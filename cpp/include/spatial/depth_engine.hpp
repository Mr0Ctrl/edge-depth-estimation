#pragma once
#include <string>
#include <memory>

namespace spatial
{
    class DepthEngine
    {
    private:
        bool m_isLoaded;
        int m_inputWidth;
        int m_inputHeight;
        struct Impl;
        std::unique_ptr<Impl> m_impl;

    public:
        DepthEngine();
        ~DepthEngine();

        bool loadModel(const std::string &modelPath, bool useGPU);
    };

} // namespace spatial
