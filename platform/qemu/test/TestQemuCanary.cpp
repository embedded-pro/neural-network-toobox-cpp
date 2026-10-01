#include <gtest/gtest.h>

namespace
{
    class TestQemuCanary
        : public ::testing::Test
    {
    protected:
        const float expected{ 1.5f };
    };
}

TEST_F(TestQemuCanary, PassingExpectationIsReported)
{
    EXPECT_FLOAT_EQ(expected, 1.5f);
}

TEST_F(TestQemuCanary, FailingExpectationIsReported)
{
    EXPECT_FLOAT_EQ(expected, 2.5f);
}
