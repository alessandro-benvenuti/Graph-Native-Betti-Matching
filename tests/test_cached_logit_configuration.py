"""Regression for the real smoke's historical config and frozen-ID retry."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from configs import ConfigError, load_config, validate_config
from experiments.cached_logits.configuration import load_source_config
from experiments.cached_logits.synthetic import config
from experiments.cached_logits.cache import select_manifest
from experiments.cached_logits.retry import smoke_cache_output


class CachedLogitConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=config()
        self.config['model']['decoder']['hidden_dim']=12
        validate_config(self.config)
        self.path=self.root/'resolved-config.yaml'

    def write(self, value):
        self.path.write_text(yaml.safe_dump(value))

    def test_modern_config_is_identical_to_production_loader(self):
        self.write(self.config)
        adapted,original,audit=load_source_config(self.path)
        self.assertEqual(adapted,load_config(self.path))
        self.assertEqual(original,self.config)
        self.assertEqual(audit['additions'],[])

    def test_reproduces_cluster_error_and_repairs_only_missing_topology_fields(self):
        old=copy.deepcopy(self.config)
        old['topology'].pop('sampling')
        old['topology'].pop('complex')
        for key in ('betti_h0','betti_h1'):
            old['topology'][key].pop('normalization')
            old['topology'][key].pop('activation_epoch')
        old['topology']['betti_h0']['unmatched_weight']=.37
        self.write(old)
        before=self.path.read_bytes()
        with self.assertRaisesRegex(ConfigError,'topology.sampling must be a mapping'):
            load_config(self.path)
        adapted,original,audit=load_source_config(self.path)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(original,old)
        for key in old:
            if key!='topology': self.assertEqual(adapted[key],old[key])
        for key in ('betti_h0','betti_h1'):
            for name,value in old['topology'][key].items():
                self.assertEqual(adapted['topology'][key][name],value)
        additions={item['path']:item['value'] for item in audit['additions']}
        self.assertFalse(additions['topology.sampling.enabled'])
        self.assertEqual(additions['topology.betti_h0.normalization'],'feature_count')
        self.assertNotIn('topology.betti_h0.unmatched_weight',additions)
        validate_config(adapted)

    def test_supplied_nondefault_topology_settings_are_preserved(self):
        old=copy.deepcopy(self.config)
        old['topology'].pop('sampling')
        old['topology']['complex'].update(aggregation='product',alpha=.7,
            max_active_unmatched=3,detach_unmatched_edge_probabilities=False)
        old['topology']['betti_h0']['normalization']='matched_mean'
        self.write(old)
        adapted,_,audit=load_source_config(self.path)
        self.assertEqual(adapted['topology']['complex'],old['topology']['complex'])
        self.assertEqual(adapted['topology']['betti_h0'],old['topology']['betti_h0'])
        self.assertTrue(all(item['path'].startswith('topology.sampling.') for item in audit['additions']))

    def test_null_schema_blocks_are_audited_but_malformed_values_are_rejected(self):
        old=copy.deepcopy(self.config)
        old['topology']['sampling']=None
        self.write(old)
        _,original,audit=load_source_config(self.path)
        self.assertIsNone(original['topology']['sampling'])
        self.assertTrue(all(item['reason']=='historical null block' for item in audit['additions']))
        old['topology']['sampling']=False
        self.write(old)
        with self.assertRaisesRegex(ConfigError,'sampling must be a mapping'):
            load_source_config(self.path)
        old['topology']['sampling']=self.config['topology']['sampling']
        old['topology']['betti_h0'].pop('unmatched_weight')
        self.write(old)
        with self.assertRaises(ConfigError): load_source_config(self.path)

    def test_frozen_manifest_reuse_preserves_ids_and_file(self):
        manifest=self.root/'smoke-manifest.json'
        value=dict(schema=1,split='train',dataset_root=str(self.root/'dataset'),
            sample_ids=['sample_000053_0885','sample_000090_0848'],
            categories={'sample_000053_0885':'bounded_pool_pending','sample_000090_0848':'bounded_pool_pending'},
            requested=2,selection_provenance=dict(seed=364505,bounded_pool=True))
        manifest.write_text(json.dumps(value))
        before=manifest.read_bytes()
        args=Namespace(output=manifest,dataset=self.root/'dataset',count=2,seed=364505,pool=None,reuse=True)
        with patch('data.loaders.discovery.discover_synthetic_mri',side_effect=AssertionError('Rescanned dataset')):
            select_manifest(args)
        self.assertEqual(manifest.read_bytes(),before)
        args.seed=1
        with self.assertRaisesRegex(ValueError,'different selection seed'): select_manifest(args)
        self.assertEqual(manifest.read_bytes(),before)
        args.seed=364505; args.reuse=False
        with self.assertRaises(FileExistsError): select_manifest(args)

    def test_smoke_retry_preserves_failed_outputs_and_refuses_completed_cache(self):
        default=self.root/'smoke-cache'
        self.assertEqual(smoke_cache_output(self.root,'123'),default)
        default.mkdir()
        evidence=default/'error.txt'; evidence.write_text('old failure')
        retry=smoke_cache_output(self.root,'124')
        self.assertEqual(retry,self.root/'smoke-cache-retry-124')
        self.assertEqual(evidence.read_text(),'old failure')
        self.assertFalse(retry.exists())
        retry.mkdir()
        with self.assertRaises(FileExistsError): smoke_cache_output(self.root,'124')
        (default/'CACHE_COMPLETE.json').write_text('{}')
        with self.assertRaisesRegex(FileExistsError,'complete smoke cache'):
            smoke_cache_output(self.root,'125')


if __name__=='__main__': unittest.main()
