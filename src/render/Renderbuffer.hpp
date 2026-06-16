#pragma once

#include "../helpers/signal/Signal.hpp"
#include "../helpers/memory/Memory.hpp"
#include "Framebuffer.hpp"
#include <aquamarine/buffer/Buffer.hpp>

namespace Render {
    class IRenderbuffer {
      public:
        IRenderbuffer(SP<Aquamarine::IBuffer> buffer, uint32_t format);
        virtual ~IRenderbuffer() = default;

        bool                    good();
        SP<IFramebuffer>        getFB();

        virtual void            bind()   = 0;
        virtual void            unbind() = 0;

        // WSL/WSLg shm present path: when the backing buffer is host memory
        // (no dmabuf), rendering goes to an offscreen FBO that must be copied
        // into the buffer's CPU mapping before presentation. Default no-op for
        // zero-copy (dmabuf) renderbuffers.
        virtual bool            isShm() { return false; }
        virtual void            readbackToBuffer() {}

        WP<Aquamarine::IBuffer> m_hlBuffer;

      protected:
        SP<IFramebuffer> m_framebuffer;
        bool             m_good = false;

        struct {
            CHyprSignalListener destroyBuffer;
        } m_listeners;
    };
}
