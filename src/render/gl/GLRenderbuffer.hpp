#pragma once

#include "../../helpers/memory/Memory.hpp"
#include "../Renderbuffer.hpp"
#include <aquamarine/buffer/Buffer.hpp>

namespace Render::GL {
    class CGLRenderbuffer : public IRenderbuffer {
      public:
        CGLRenderbuffer(SP<Aquamarine::IBuffer> buffer, uint32_t format);
        ~CGLRenderbuffer();

        void bind() override;
        void unbind() override;

        // In WSL/WSLg shm mode the underlying buffer is host memory with no
        // dmabuf, so we render into an offscreen renderbuffer and copy the
        // result into the buffer's CPU mapping before it is presented.
        bool isShm() override;
        void readbackToBuffer() override;

      private:
        void*  m_image = nullptr;
        GLuint m_rbo   = 0;
        bool   m_shm   = false;
    };
}
