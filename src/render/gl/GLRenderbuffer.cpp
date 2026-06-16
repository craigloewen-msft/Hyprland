#include "GLRenderbuffer.hpp"
#include "../Renderer.hpp"
#include "../OpenGL.hpp"
#include "../../Compositor.hpp"
#include "../Framebuffer.hpp"
#include "GLFramebuffer.hpp"
#include "../Renderbuffer.hpp"
#include <hyprgraphics/egl/Egl.hpp>
#include <hyprutils/memory/SharedPtr.hpp>
#include <hyprutils/memory/Casts.hpp>
#include <hyprutils/signal/Listener.hpp>
#include <hyprutils/signal/Signal.hpp>

#include <dlfcn.h>

using namespace Render::GL;
using namespace Hyprgraphics::Egl;

CGLRenderbuffer::~CGLRenderbuffer() {
    if (!g_pCompositor || g_pCompositor->m_isShuttingDown || !g_pHyprRenderer)
        return;

    g_pHyprOpenGL->makeEGLCurrent();

    unbind();
    m_framebuffer->release();

    if (m_rbo)
        glDeleteRenderbuffers(1, &m_rbo);

    if (m_image != EGL_NO_IMAGE_KHR)
        g_pHyprOpenGL->m_proc.eglDestroyImageKHR(g_pHyprOpenGL->m_eglDisplay, m_image);
}

CGLRenderbuffer::CGLRenderbuffer(SP<Aquamarine::IBuffer> buffer, uint32_t format) : IRenderbuffer(buffer, format) {
    auto dma = buffer->dmabuf();

    if (!dma.success) {
        // WSL/WSLg: the buffer is host memory (wl_shm) with no dmabuf. Render
        // into a plain offscreen renderbuffer; readbackToBuffer() copies the
        // result into the buffer's CPU mapping before it is presented.
        m_shm = true;

        glGenRenderbuffers(1, &m_rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, buffer->size.x, buffer->size.y);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        m_framebuffer = makeShared<CGLFramebuffer>();
        glGenFramebuffers(1, &GLFB(m_framebuffer)->m_fb);
        GLFB(m_framebuffer)->m_fbAllocated = true;
        m_framebuffer->m_size              = buffer->size;
        m_framebuffer->m_drmFormat         = format;
        m_framebuffer->bind();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_rbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            Log::logger->log(Log::ERR, "rbo(shm): glCheckFramebufferStatus failed");
            return;
        }

        GLFB(m_framebuffer)->unbind();

        m_listeners.destroyBuffer = buffer->events.destroy.listen([this] { g_pHyprRenderer->onRenderbufferDestroy(this); });

        m_good = true;
        return;
    }

    m_image = g_pHyprOpenGL->createEGLImage(dma);
    if (m_image == EGL_NO_IMAGE_KHR) {
        Log::logger->log(Log::ERR, "rb: createEGLImage failed");
        return;
    }

    glGenRenderbuffers(1, &m_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
    g_pHyprOpenGL->m_proc.glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, m_image);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    m_framebuffer = makeShared<CGLFramebuffer>();
    glGenFramebuffers(1, &GLFB(m_framebuffer)->m_fb);
    GLFB(m_framebuffer)->m_fbAllocated = true;
    m_framebuffer->m_size              = buffer->size;
    m_framebuffer->m_drmFormat         = dma.format;
    m_framebuffer->bind();
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        Log::logger->log(Log::ERR, "rbo: glCheckFramebufferStatus failed");
        return;
    }

    GLFB(m_framebuffer)->unbind();

    m_listeners.destroyBuffer = buffer->events.destroy.listen([this] { g_pHyprRenderer->onRenderbufferDestroy(this); });

    m_good = true;
}

void CGLRenderbuffer::bind() {
    g_pHyprOpenGL->makeEGLCurrent();
    g_pHyprRenderer->bindFB(m_framebuffer);
}

void CGLRenderbuffer::unbind() {
    GLFB(m_framebuffer)->unbind();
}

bool CGLRenderbuffer::isShm() {
    return m_shm;
}

void CGLRenderbuffer::readbackToBuffer() {
    if (!m_shm || !m_good)
        return;

    auto buffer = m_hlBuffer.lock();
    if (!buffer)
        return;

    const auto shm                 = buffer->shm();
    auto [pixelData, fmt, bufLen]  = buffer->beginDataPtr(0); // shm: endDataPtr is a no-op
    if (!pixelData)
        return;

    const auto PFORMAT = getPixelFormatFromDRM(shm.format);
    if (!PFORMAT) {
        Log::logger->log(Log::ERR, "rbo(shm): readback couldn't find a pixel format for 0x{:x}", shm.format);
        return;
    }

    g_pHyprOpenGL->makeEGLCurrent();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, GLFB(m_framebuffer)->getFBID());

    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // The shm buffer is ARGB8888/XRGB8888 (DRM fourcc), i.e. BGRA byte order in
    // little-endian memory, so read directly as BGRA when the format calls for
    // it to avoid a red/blue swap. Mirrors CGLFramebuffer::readPixels.
    int         glFormat          = PFORMAT->glFormat;
    static auto stripSwizzleAlpha = [](std::array<GLint, 4> arr) {
        arr[3] = GL_ONE;
        return arr;
    };

    if (PFORMAT->swizzle.has_value()) {
        if (stripSwizzleAlpha(*PFORMAT->swizzle) == stripSwizzleAlpha(SWIZZLE_RGBA))
            glFormat = GL_RGBA;
        else if (stripSwizzleAlpha(*PFORMAT->swizzle) == stripSwizzleAlpha(SWIZZLE_BGRA))
            glFormat = GL_BGRA_EXT;
        else {
            Log::logger->log(Log::ERR, "rbo(shm): unexpected swizzle, colors may be flipped");
            glFormat = GL_RGBA;
        }
    } else if (glFormat == GL_RGBA)
        glFormat = GL_BGRA_EXT;

    const auto     WIDTH      = sc<int>(m_framebuffer->m_size.x);
    const auto     HEIGHT     = sc<int>(m_framebuffer->m_size.y);
    const uint32_t packStride = minStride(PFORMAT, WIDTH);

    if (packStride == sc<uint32_t>(shm.stride))
        glReadPixels(0, 0, WIDTH, HEIGHT, glFormat, PFORMAT->glType, pixelData);
    else {
        for (int y = 0; y < HEIGHT; ++y)
            glReadPixels(0, y, WIDTH, 1, glFormat, PFORMAT->glType, pixelData + sc<size_t>(y) * shm.stride);
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}
