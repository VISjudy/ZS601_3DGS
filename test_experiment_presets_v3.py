import unittest

from experiment_presets_v3 import (
    EXPERIMENT_PRESETS,
    FEATURE_NAMES,
    PREPROCESS_FEATURES,
    TRAINING_FEATURES,
    resolve_experiment_config,
    resolve_feature_flags,
)


class ExperimentPresetTests(unittest.TestCase):
    def test_every_preset_defines_every_feature(self):
        expected=set(FEATURE_NAMES)
        for group,preset in EXPERIMENT_PRESETS.items():
            with self.subTest(group=group):
                self.assertEqual(expected,set(preset))
                self.assertTrue(all(isinstance(v,bool) for v in preset.values()))

    def test_original_disables_new_training_and_preprocess_features(self):
        flags=resolve_feature_flags("original")
        self.assertTrue(flags["validation_diagnostics"])
        self.assertFalse(any(v for k,v in flags.items() if k!="validation_diagnostics"))

    def test_custom_disables_all_features(self):
        self.assertFalse(any(resolve_feature_flags("custom").values()))

    def test_existing_group_semantics(self):
        a=resolve_feature_flags("A",feature_names=TRAINING_FEATURES)
        b=resolve_feature_flags("B",feature_names=TRAINING_FEATURES)
        c=resolve_feature_flags("C",feature_names=TRAINING_FEATURES)
        d=resolve_feature_flags("D",feature_names=TRAINING_FEATURES)
        e=resolve_feature_flags("E",feature_names=TRAINING_FEATURES)
        self.assertTrue(a["init_flatten"])
        self.assertFalse(a["surface_loss"])
        self.assertTrue(b["surface_loss"])
        self.assertTrue(c["scale_bounds"])
        self.assertTrue(d["surface_densify"])
        self.assertFalse(d["lidar_depth_loss"])
        self.assertTrue(e["lidar_depth_loss"])
        self.assertFalse(e["surface_densify"])

    def test_auto_inherits_and_explicit_values_override(self):
        flags=resolve_feature_flags("A",{
            "validation_diagnostics":"off",
            "lidar_depth_loss":"on",
            "init_normal":"auto",
        },TRAINING_FEATURES)
        self.assertFalse(flags["validation_diagnostics"])
        self.assertTrue(flags["lidar_depth_loss"])
        self.assertEqual(EXPERIMENT_PRESETS["A"]["init_normal"],flags["init_normal"])

    def test_group_letters_case_insensitive(self):
        self.assertEqual(resolve_feature_flags("c"),resolve_feature_flags("C"))

    def test_preprocess_flags_are_independently_resolved(self):
        flags=resolve_feature_flags("custom",{
            "pca_normal_estimation":"on",
            "camera_normal_orientation":True,
        },PREPROCESS_FEATURES)
        self.assertTrue(flags["pca_normal_estimation"])
        self.assertTrue(flags["camera_normal_orientation"])
        self.assertFalse(flags["lidar_outlier_filter"])

    def test_a_through_e_enable_standard_preprocessing_only(self):
        expected_on = {
            "pca_normal_estimation", "camera_normal_orientation",
            "supervision_generation", "visibility_occlusion_check",
        }
        for group in "ABCDE":
            with self.subTest(group=group):
                flags = resolve_feature_flags(group, feature_names=PREPROCESS_FEATURES)
                self.assertEqual(expected_on, {name for name, enabled in flags.items() if enabled})

    def test_config_records_only_enabled_parameters(self):
        config=resolve_experiment_config("custom",{"normal_loss":"on"},{
            "normal_loss":{"weight":0.05},
            "size_loss":{"weight":0.01},
        },TRAINING_FEATURES)
        self.assertEqual({"normal_loss":{"weight":0.05}},
                         config["enabled_feature_params"])

    def test_invalid_feature_and_value_rejected(self):
        with self.assertRaises(ValueError):
            resolve_feature_flags("A",{"not_a_feature":"on"},TRAINING_FEATURES)
        with self.assertRaises(ValueError):
            resolve_feature_flags("A",{"size_loss":"yes"},TRAINING_FEATURES)
        with self.assertRaises(ValueError):
            resolve_feature_flags("unknown")

    def test_results_do_not_mutate_presets(self):
        flags=resolve_feature_flags("A")
        flags["validation_diagnostics"]=False
        self.assertTrue(EXPERIMENT_PRESETS["A"]["validation_diagnostics"])


if __name__=="__main__":
    unittest.main()
