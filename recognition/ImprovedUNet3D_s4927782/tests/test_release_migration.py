"""Synthetic documentary-migration checks; no images or model inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dataset as data
import train
from tools import package_release as package
from utils import load_json, save_json, sha256_file


def source_pair(folder):
    old, current = folder/'execution', folder/'deployment'
    old.mkdir(); current.mkdir()
    for name in train.SOURCE_FILES:
        original = '"""Execution documentation."""\n\ndef value():\n    """Function documentation."""\n    return 7\n'
        (old/name).write_text(original)
        (current/name).write_text(original.replace('Execution documentation.', 'Delivery documentation.')
                                  if name in package.DOCUMENTARY_SOURCE_FILES else original)
    config = {'epochs': 30, 'seed': 3711, 'lr': .001, 'evaluation_context': 'Execution description.'}
    save_json(old/'config.json', config)
    config['evaluation_context'] = 'Patient-disjoint study; selected weights frozen before final TEST evaluation.'
    save_json(current/'config.json', config)
    return old, current


def synthetic_checkpoint(execution_config, proof):
    history = []
    for epoch in range(1, 31):
        score = epoch/40 if epoch in [1, 5, 10, 15, 20, 25, 30] else None
        history.append({'epoch': epoch, 'train_loss': 1/epoch, 'validation_prostate_dice': score,
                        'validation': {'configuration': deepcopy(execution_config), 'score': score}
                        if score is not None else None})
    return {'schema_version': 1, 'model': 'unet2d', 'config': execution_config,
            'identity': {'manifest_sha256': 'a'*64, 'trainval_receipt_sha256': 'b'*64,
                         'config_sha256': data.config_digest(execution_config),
                         'source_sha256': {n: r['execution_sha256'] for n, r in proof['source_files'].items()}},
            'runtime': {'source_commit': 'synthetic_fixture'},
            'state_dict': {'weight': torch.tensor([1., 2.])},
            'epoch': 30, 'best_epoch': 30, 'best_score': .75, 'history': history,
            'optimizer': {}, 'scaler': {}}


class DocumentaryMigrationTests(unittest.TestCase):
    def test_documentary_change_preserves_execution_identity_and_current_strict_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            old = folder/'execution'
            old.mkdir()
            current_config = load_json(ROOT/'config.json')
            old_config = deepcopy(current_config)
            old_config['evaluation_context'] = 'Synthetic execution description.'
            for name in train.SOURCE_FILES:
                encoded = (ROOT/name).read_text()
                if name in package.DOCUMENTARY_SOURCE_FILES:
                    encoded = encoded.replace('"""', '"""Synthetic documentation.\n', 1)
                (old/name).write_text(encoded)
            save_json(old/'config.json', old_config)
            execution_config, proof = package.documentation_proof(old, ROOT)
            original = synthetic_checkpoint(execution_config, proof)
            original_path = folder/'original.pt'
            torch.save(original, original_path)
            original_sha = sha256_file(original_path)
            deployed = package.migrate_checkpoint(original, current_config, train.source_identity(), proof, original_sha)
            package.verify_documentary_checkpoint(original, deployed, proof, original_sha)
            self.assertEqual(deployed['execution_identity'], original['identity'])
            self.assertEqual(deployed['runtime'], original['runtime'])
            self.assertEqual(deployed['documentation_migration']['original_full_history_sha256'], package.json_digest(original['history']))
            self.assertEqual(len([p for p in deployed['documentation_migration']['changed_checkpoint_paths'] if p.startswith('history/')]), 7)
            self.assertEqual(original['history'][0]['validation']['configuration']['evaluation_context'], 'Synthetic execution description.')
            self.assertEqual(deployed['history'][0]['validation']['configuration']['evaluation_context'], current_config['evaluation_context'])
            path = folder/'inference.pt'
            torch.save(deployed, path)
            loaded = train.load_selected_checkpoint(path, current_config)
            self.assertTrue(torch.equal(loaded['state_dict']['weight'], original['state_dict']['weight']))
            # The strict source check still rejects the untouched execution
            # checkpoint when presented as a current deployment checkpoint.
            with self.assertRaisesRegex(ValueError, 'sources differ|legacy checkpoint file bytes differ'):
                train.load_selected_checkpoint(original_path, execution_config)

    def test_executable_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, current = source_pair(Path(temporary))
            path = current/'modules.py'
            path.write_text(path.read_text().replace('return 7', 'return 8'))
            with self.assertRaisesRegex(ValueError, 'Executable AST changed'):
                package.documentation_proof(old, current)

    def test_function_docstring_and_immutable_source_edits_are_rejected(self):
        for filename, old_text, new_text in [('modules.py', 'Function documentation.', 'Changed function doc.'),
                                             ('predict.py', 'Execution documentation.', 'Changed module doc.')]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                old, current = source_pair(Path(temporary))
                path = current/filename
                path.write_text(path.read_text().replace(old_text, new_text))
                with self.assertRaises(ValueError):
                    package.documentation_proof(old, current)

    def test_numerical_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, current = source_pair(Path(temporary))
            config = load_json(current/'config.json')
            config['lr'] = .002
            save_json(current/'config.json', config)
            with self.assertRaisesRegex(ValueError, 'config excluding evaluation_context'):
                package.documentation_proof(old, current)

    def test_tensor_numeric_history_and_undeclared_metadata_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, current = source_pair(Path(temporary))
            execution_config, proof = package.documentation_proof(old, current)
            original = synthetic_checkpoint(execution_config, proof)
            source_sha = {n: r['deployment_sha256'] for n, r in proof['source_files'].items()}
            clean = package.migrate_checkpoint(original, load_json(current/'config.json'), source_sha, proof, 'c'*64)
            for change in ('tensor', 'score', 'identity', 'runtime', 'context_path'):
                changed = deepcopy(clean)
                if change == 'tensor':
                    changed['state_dict']['weight'][0] += 1
                elif change == 'score':
                    changed['history'][0]['validation_prostate_dice'] += .01
                elif change == 'identity':
                    changed['identity']['manifest_sha256'] = 'd'*64
                elif change == 'runtime':
                    changed['runtime']['source_commit'] = 'different'
                else:
                    changed['documentation_migration']['changed_checkpoint_paths'].append('history/0/train_loss')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    package.verify_documentary_checkpoint(original, changed, proof, 'c'*64)


if __name__ == '__main__':
    unittest.main()
