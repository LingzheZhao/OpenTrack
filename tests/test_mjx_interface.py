"""The MJX interface track_mj.utils.mjx_backend calls, read on the installed MJX without physics."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import inspect
import unittest

from mujoco import mjx


class MjxInterfaceTest(unittest.TestCase):
    def test_put_model_and_make_data_accept_the_arguments_the_helper_passes(self):
        put_model = inspect.signature(mjx.put_model).parameters
        make_data = inspect.signature(mjx.make_data).parameters
        self.assertTrue({"impl", "graph_mode"} <= set(put_model))
        self.assertTrue({"impl", "naconmax", "naccdmax", "njmax"} <= set(make_data))

    def test_warp_graph_modes_and_contact_fields_exist(self):
        from mujoco.mjx.warp import types as mjxw_types

        self.assertTrue(hasattr(mjxw_types.GraphMode, "WARP"))
        fields = set(mjxw_types.DataWarp.__annotations__)
        for name in ("contact__geom", "contact__dist", "contact__frame", "contact__worldid", "nacon", "overflow"):
            self.assertIn(name, fields)
        # The PPO pmap axes keep these leaves unmapped (train_tracking._warp_non_vmappable_pmap_axes).
        self.assertTrue({"contact__geom", "contact__worldid"} <= set(mjxw_types.DATA_NON_VMAP))

    def test_impl_enum_names(self):
        self.assertEqual({mjx.Impl.JAX.value, mjx.Impl.WARP.value}, {"jax", "warp"})


if __name__ == "__main__":
    unittest.main()
