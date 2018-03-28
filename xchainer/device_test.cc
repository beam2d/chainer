#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/native/native_device.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override { context_session_.emplace(); }

    void TearDown() override { context_session_.reset(); }

private:
    nonstd::optional<testing::ContextSession> context_session_;
};

TEST_F(DeviceTest, Ctor) {
    Context& ctx = GetDefaultContext();
    native::NativeBackend native_backend{ctx};
    {
        native::NativeDevice device{native_backend, 0};
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        native::NativeDevice device{native_backend, 1};
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST_F(DeviceTest, SetDefaultDevice) {
    Context& ctx = GetDefaultContext();
    EXPECT_EQ(&ctx.GetDevice({"native", 0}), &GetDefaultDevice());

    native::NativeBackend native_backend{ctx};
    native::NativeDevice native_device{native_backend, 0};
    SetDefaultDevice(&native_device);
    ASSERT_EQ(&native_device, &GetDefaultDevice());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};
    SetDefaultDevice(&cuda_device);
    ASSERT_EQ(&cuda_device, &GetDefaultDevice());
#endif  // XCHAINER_ENABLE_CUDA

    native::NativeBackend native_backend2{ctx};
    native::NativeDevice native_device2{native_backend2, 2};
    SetDefaultDevice(&native_device2);
    ASSERT_EQ(&native_device2, &GetDefaultDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    Context& ctx = GetDefaultContext();
    native::NativeBackend backend1{ctx};
    native::NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);

    native::NativeBackend backend2{ctx};
    native::NativeDevice device2{backend2, 2};
    auto future = std::async(std::launch::async, [&ctx, &device2] {
        SetDefaultContext(&ctx);
        SetDefaultDevice(&device2);
        return &GetDefaultDevice();
    });
    ASSERT_NE(&GetDefaultDevice(), future.get());
}

TEST_F(DeviceTest, ThreadLocalDefault) {
    Context& ctx = GetDefaultContext();
    SetGlobalDefaultContext(&ctx);
    auto reset_global = gsl::finally([] { SetGlobalDefaultContext(nullptr); });

    Device& device = ctx.GetDevice({"native", 0});

    native::NativeBackend backend1{ctx};
    native::NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);
    auto future = std::async(std::launch::async, [] { return &GetDefaultDevice(); });

    EXPECT_EQ(&device, future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    Context& ctx = GetDefaultContext();
    {
        // DeviceScope should work even if default device is not set
        native::NativeBackend backend{ctx};
        native::NativeDevice device{backend, 0};
        DeviceScope scope(device);
    }
    native::NativeBackend backend1{ctx};
    native::NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);
    {
        native::NativeBackend backend2{ctx};
        native::NativeDevice device2{backend2, 2};
        DeviceScope scope(device2);
        EXPECT_EQ(&device2, &GetDefaultDevice());
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(&device1, &GetDefaultDevice());
        native::NativeBackend backend2{ctx};
        native::NativeDevice device2{backend2, 2};
        SetDefaultDevice(&device2);
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    native::NativeBackend backend2{ctx};
    native::NativeDevice device2{backend2, 2};
    {
        DeviceScope scope(device2);
        scope.Exit();
        EXPECT_EQ(&device1, &GetDefaultDevice());
        SetDefaultDevice(&device2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(&device2, &GetDefaultDevice());
}

}  // namespace
}  // namespace xchainer
