import importlib.util,json,unittest,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/run_experiment.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class Reproduction(unittest.TestCase):
    def test_trainer_matches_frozen_live_source(self):runner.verify_code()
    def test_all_presets_share_budget_and_only_expected_losses(self):
        expected={'A':(False,False,0),'B':(True,False,0),'C':(False,True,0),'D':(True,True,0),'E':(True,True,200),'F-real':(True,True,200)}
        for arm,flags in expected.items():
            c=runner.read_config(arm);self.assertEqual((c['depth_loss'],c['normal_loss'],c['virtual_views']),flags)
            self.assertEqual(c['training']['iterations'],50000);self.assertEqual(c['training']['position_lr_max_steps'],150000)
    def test_command_uses_original_optimizer_and_explicit_paths(self):
        inputs={k:'/input/'+k for k in ['source_path','point_cloud','train_file','val_file','test_file','cameras_file','supervision_root','supervision_manifest']}
        cmd=runner.build_command(runner.read_config('C'),inputs,'/drive/fresh')
        self.assertEqual(cmd[cmd.index('--paper_arm')+1],'C');self.assertEqual(cmd[cmd.index('--experiment')+1],'original')
        self.assertEqual(cmd[cmd.index('--model_path')+1],'/drive/fresh')
    def test_missing_data_rejected_before_training(self):
        with self.assertRaises(AssertionError):runner.preflight(runner.read_config('A'),{'source_path':'/nonexistent-v018-data'})

if __name__=='__main__':unittest.main()
