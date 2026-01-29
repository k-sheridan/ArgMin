"""
Basic tests for tangent_py.
"""
import pytest
import tangent_py as tg


class TestInitialization:
    def test_init_succeeds(self):
        """Test that initialization works."""
        tg.init()
        assert tg.is_initialized()

    def test_init_idempotent(self):
        """Test that init can be called multiple times."""
        tg.init()
        tg.init()
        assert tg.is_initialized()


class TestVariables:
    def test_simple_scalar_creation(self):
        """Test SimpleScalar variable creation."""
        tg.init()
        x = tg.Tangent.SimpleScalar(5.0)
        assert x.value == 5.0

    def test_simple_scalar_default(self):
        """Test SimpleScalar default value."""
        tg.init()
        x = tg.Tangent.SimpleScalar()
        assert x.value == 0.0

    def test_inverse_depth_creation(self):
        """Test InverseDepth variable creation."""
        tg.init()
        d = tg.Tangent.InverseDepth(0.5)
        assert d.value == 0.5

    def test_se3_creation(self):
        """Test SE3 variable creation (identity)."""
        tg.init()
        pose = tg.Tangent.SE3()
        # Check it's identity (translation should be zero)
        assert pose.value.translation()[0] == 0.0
        assert pose.value.translation()[1] == 0.0
        assert pose.value.translation()[2] == 0.0


class TestErrorTermDefinition:
    def test_define_simple_error(self):
        """Test defining a simple error term."""
        tg.init()

        tg.define_error_term("""
        class TestDiffError : public Tangent::AutoDiffErrorTerm<
            TestDiffError, double, 1,
            Tangent::SimpleScalar, Tangent::SimpleScalar> {
        public:
            TestDiffError(Tangent::VariableKey<Tangent::SimpleScalar> k1,
                          Tangent::VariableKey<Tangent::SimpleScalar> k2) {
                std::get<0>(variableKeys) = k1;
                std::get<1>(variableKeys) = k2;
                information.setIdentity();
            }

            template <typename T, typename V1, typename V2>
            Eigen::Matrix<T, 1, 1> computeError(const V1& v1, const V2& v2) const {
                Eigen::Matrix<T, 1, 1> err;
                err(0) = v2 - v1;
                return err;
            }
        };
        """)

        # Verify class exists
        import cppyy
        assert hasattr(cppyy.gbl, "TestDiffError")


class TestOptimization:
    def test_simple_scalar_optimization(self):
        """Test optimizing two scalars to be equal."""
        tg.init()

        # Define error term
        tg.define_error_term("""
        class ScalarDiffError : public Tangent::AutoDiffErrorTerm<
            ScalarDiffError, double, 1,
            Tangent::SimpleScalar, Tangent::SimpleScalar> {
        public:
            ScalarDiffError(Tangent::VariableKey<Tangent::SimpleScalar> k1,
                            Tangent::VariableKey<Tangent::SimpleScalar> k2) {
                std::get<0>(variableKeys) = k1;
                std::get<1>(variableKeys) = k2;
                information.setIdentity();
            }

            template <typename T, typename V1, typename V2>
            Eigen::Matrix<T, 1, 1> computeError(const V1& v1, const V2& v2) const {
                Eigen::Matrix<T, 1, 1> err;
                err(0) = v2 - v1;
                return err;
            }
        };
        """)

        # Create optimizer
        opt = tg.Optimizer(
            variables=["SimpleScalar"],
            error_terms=["ScalarDiffError"]
        )

        # Add variables: x=10, y=0 with strong prior on y
        x = tg.Tangent.SimpleScalar(10.0)
        y = tg.Tangent.SimpleScalar(0.0)

        k1 = opt.add_variable(x)
        k2 = opt.add_variable(y)
        opt.set_prior(k2, 1e12)  # Strong prior keeps y at 0

        # Add error term: we want x == y
        opt.add_error_term("ScalarDiffError", k1, k2)

        # Optimize
        result = opt.optimize()

        # Check convergence
        assert result.error_decreased

        # x should have moved toward y (which is held at 0)
        final_x = opt.get_variable(k1).value
        final_y = opt.get_variable(k2).value

        assert abs(final_x - final_y) < 0.1  # Should be nearly equal
        assert abs(final_y) < 0.01  # y should stay near 0


class TestTemplates:
    def test_error_term_template(self):
        """Test the error_term_template helper."""
        code = tg.error_term_template(
            name="TemplatedDiffError",
            residual_dim=1,
            var_types=["SimpleScalar", "SimpleScalar"],
            compute_body="err(0) = v1 - v0;"
        )

        assert "class TemplatedDiffError" in code
        assert "AutoDiffErrorTerm" in code
        assert "err(0) = v1 - v0;" in code

    def test_error_term_template_with_extra_members(self):
        """Test error_term_template with extra members (prior pattern)."""
        code = tg.error_term_template(
            name="ScalarPriorError",
            residual_dim=1,
            var_types=["SimpleScalar"],
            compute_body="err(0) = v0 - target;",
            extra_members="double target;",
            extra_constructor_params="double t",
            extra_constructor_init="target = t;"
        )

        assert "class ScalarPriorError" in code
        assert "double target;" in code
        assert "double t" in code

    def test_template_with_define(self):
        """Test that generated template code can be compiled."""
        tg.init()

        # Generate code using template
        code = tg.error_term_template(
            name="GeneratedDiffError",
            residual_dim=1,
            var_types=["SimpleScalar", "SimpleScalar"],
            compute_body="err(0) = v1 - v0;"
        )

        # Define it
        tg.define_error_term(code)

        # Verify it exists
        import cppyy
        assert hasattr(cppyy.gbl, "GeneratedDiffError")
