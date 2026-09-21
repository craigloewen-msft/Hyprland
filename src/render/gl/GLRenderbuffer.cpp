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
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace Render::GL;
using namespace Hyprgraphics::Egl;

namespace {
    constexpr uint32_t BRIDGE_MAGIC   = 0x4e49574b;
    constexpr uint32_t BRIDGE_VERSION = 2;

    struct alignas(8) SBridgeHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t width;
        uint32_t height;
        uint32_t stride;
        uint32_t format;
        uint32_t damageX;
        uint32_t damageY;
        uint32_t damageWidth;
        uint32_t damageHeight;
        uint64_t payloadSize;
        uint64_t sequence;
    };

    static_assert(sizeof(SBridgeHeader) == 56);

    class CFramebufferBridge {
      public:
        ~CFramebufferBridge() {
            close();
        }

        bool ensure(uint32_t width, uint32_t height, uint32_t format) {
            const char* path = std::getenv("WSLG_RDP_FRAMEBUFFER");
            if (!path || !*path)
                path = std::getenv("WSLG_KWIN_FRAMEBUFFER");
            if (!path || !*path) {
                close();
                return false;
            }

            const uint32_t stride       = width * 4;
            const size_t   payloadSize  = sc<size_t>(stride) * height;
            const size_t   requiredSize = sizeof(SBridgeHeader) + payloadSize;
            if (m_mapping && m_path == path && m_mappingSize == requiredSize)
                return true;

            close();
            const int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd < 0) {
                Log::logger->log(Log::ERR, "rbo(shm): failed to create WSLg framebuffer bridge {}: {}", path, strerror(errno));
                return false;
            }

            if (ftruncate(fd, requiredSize) != 0) {
                Log::logger->log(Log::ERR, "rbo(shm): failed to size WSLg framebuffer bridge {}: {}", path, strerror(errno));
                ::close(fd);
                return false;
            }

            void* mapping = mmap(nullptr, requiredSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            ::close(fd);
            if (mapping == MAP_FAILED) {
                Log::logger->log(Log::ERR, "rbo(shm): failed to map WSLg framebuffer bridge {}: {}", path, strerror(errno));
                return false;
            }

            m_mapping     = mapping;
            m_mappingSize = requiredSize;
            m_path        = path;
            m_width       = width;
            m_height      = height;
            m_stride      = stride;
            m_format      = format;
            m_needsFull   = true;
            memset(m_mapping, 0, m_mappingSize);
            return true;
        }

        bool needsFullFrame() const {
            return m_needsFull;
        }

        void publish(uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t* pixels) {
            if (!m_mapping || !width || !height)
                return;

            auto*             header = sc<SBridgeHeader*>(m_mapping);
            std::atomic_ref   sequence(header->sequence);
            const uint64_t    writeSequence = (sequence.load(std::memory_order_relaxed) + 1) | 1;
            const size_t      rowBytes      = sc<size_t>(width) * 4;
            auto*             framebuffer   = sc<uint8_t*>(m_mapping) + sizeof(SBridgeHeader);

            sequence.store(writeSequence, std::memory_order_release);
            header->magic        = BRIDGE_MAGIC;
            header->version      = BRIDGE_VERSION;
            header->width        = m_width;
            header->height       = m_height;
            header->stride       = m_stride;
            header->format       = m_format;
            header->damageX      = x;
            header->damageY      = y;
            header->damageWidth  = width;
            header->damageHeight = height;
            header->payloadSize  = sc<uint64_t>(m_stride) * m_height;

            for (uint32_t row = 0; row < height; ++row) {
                // Hyprland's output projection already accounts for OpenGL's
                // framebuffer origin, so glReadPixels returns logical top-down
                // rows for this offscreen output.
                const auto* source = pixels + sc<size_t>(row) * rowBytes;
                auto* destination = framebuffer + sc<size_t>(y + row) * m_stride + sc<size_t>(x) * 4;
                memcpy(destination, source, rowBytes);
            }

            sequence.store(writeSequence + 1, std::memory_order_release);
            m_needsFull = false;
        }

      private:
        void close() {
            if (m_mapping)
                munmap(m_mapping, m_mappingSize);
            m_mapping     = nullptr;
            m_mappingSize = 0;
            m_path.clear();
            m_width     = 0;
            m_height    = 0;
            m_stride    = 0;
            m_format    = 0;
            m_needsFull = true;
        }

        void*       m_mapping     = nullptr;
        size_t      m_mappingSize = 0;
        std::string m_path;
        uint32_t    m_width       = 0;
        uint32_t    m_height      = 0;
        uint32_t    m_stride      = 0;
        uint32_t    m_format      = 0;
        bool        m_needsFull   = true;
    };

    CFramebufferBridge g_framebufferBridge;
}

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

void CGLRenderbuffer::readbackToBuffer(const CRegion& damage) {
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

    const int WIDTH  = sc<int>(m_framebuffer->m_size.x);
    const int HEIGHT = sc<int>(m_framebuffer->m_size.y);

    const bool bridgeReady = g_framebufferBridge.ensure(WIDTH, HEIGHT, shm.format);
    const auto extents     = damage.pixman()->extents;
    int        x           = std::clamp(sc<int>(extents.x1), 0, WIDTH);
    int        y           = std::clamp(sc<int>(extents.y1), 0, HEIGHT);
    int        right       = std::clamp(sc<int>(extents.x2), x, WIDTH);
    int        bottom      = std::clamp(sc<int>(extents.y2), y, HEIGHT);

    if (bridgeReady && g_framebufferBridge.needsFullFrame()) {
        x      = 0;
        y      = 0;
        right  = WIDTH;
        bottom = HEIGHT;
    }

    const int readWidth  = right - x;
    const int readHeight = bottom - y;
    if (readWidth > 0 && readHeight > 0) {
        const uint32_t   rowBytes = minStride(PFORMAT, readWidth);
        std::vector<uint8_t> readback(sc<size_t>(rowBytes) * readHeight);
        // Hyprland's output projection maps logical top-left coordinates
        // directly into this offscreen framebuffer, including for subregions.
        const int        glY      = y;

        glReadPixels(x, glY, readWidth, readHeight, glFormat, PFORMAT->glType, readback.data());

        for (int row = 0; row < readHeight; ++row) {
            auto* destination = pixelData + sc<size_t>(glY + row) * shm.stride + sc<size_t>(x) * 4;
            memcpy(destination, readback.data() + sc<size_t>(row) * rowBytes, rowBytes);
        }

        if (bridgeReady)
            g_framebufferBridge.publish(x, y, readWidth, readHeight, readback.data());
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}
