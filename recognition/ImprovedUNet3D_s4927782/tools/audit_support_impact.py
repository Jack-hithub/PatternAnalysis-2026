"""Quantify a raw-support inference correction without running either model.

Raw MRI and Boolean sidecars remain private. The public aggregate contains
counts and source/checkpoint identities only. Immutable old caches stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dataset as corrected
from utils import load_json,save_json,sha256_file


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(args):
    out=Path(args.output_dir).resolve()
    if out.exists():raise FileExistsError('Use a new audit directory')
    out.mkdir(parents=True);(out/'support').mkdir()
    legacyroot=Path(args.legacy_source_dir).resolve()
    package=load_json(legacyroot/'PACKAGE_MANIFEST.json')
    expected=package['deployment_source_sha256']
    for name,digest in expected.items():
        if sha256_file(legacyroot/name)!=digest:raise ValueError('Legacy source differs: '+name)
    spec=importlib.util.spec_from_file_location('frozen_support_audit_dataset',legacyroot/'dataset.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    manifest_path=Path(args.manifest).resolve();manifest=legacy.load_manifest(manifest_path)
    test=load_json(manifest_path.parent/'test_completion_receipt.json')
    prepared={(r['patient'],r['visit']):r for r in manifest['records'] if r['split']!='test'}
    prepared.update({(r['patient'],r['visit']):r for r in test['records']})
    if len(prepared)!=211:raise ValueError('Expected all211 original case identities')
    raw=Path(args.raw_root).resolve()
    counters=lambda:{key:0 for key in ('cases','valid_zero_voxels','outside_support_voxels','changed_mask_voxels','raw_zero_voxels','total_voxels','cached_images_checked')}
    groups={name:counters() for name in legacy.SPLITS}
    records=[];nonidentical_cache_inputs=[];started=time.time()
    for i,record in enumerate(manifest['records'],1):
        original=prepared[(record['patient'],record['visit'])]
        for role,key in [('image','source_image'),('mask','source_mask')]:
            path=legacy._safe(raw,original[key])
            if sha256_file(path)!=original['source_sha256'][role]:raise ValueError('Raw bytes differ: '+original[key])
        image_path=legacy._safe(raw,original['source_image'])
        old_image,metadata=legacy.preprocess_image(image_path,manifest['target_shape_dhw'])
        new_image,support,new_metadata=corrected.preprocess_image(image_path,manifest['target_shape_dhw'],return_support=True)
        if not np.array_equal(old_image,new_image):raise ValueError('Normalized image changed: '+record['patient']+'_'+record['visit'])
        native=nib.as_closest_canonical(nib.load(str(image_path)))
        raw_values=native.get_fdata(dtype=np.float32,caching='unchanged')
        _,factors=legacy._grid(native.affine,native.shape,manifest['target_shape_dhw'])
        independent=zoom((raw_values!=0).astype(np.uint8),factors,order=0,grid_mode=True,mode='nearest',prefilter=False).astype(bool).transpose(2,1,0)
        if support.dtype!=np.bool_ or not np.array_equal(support,independent):raise ValueError('Explicit raw support mismatch')
        if np.any(old_image[~support]!=0):raise ValueError('Legacy image has nonzero outside true raw support')
        case_id=record['patient']+'_'+record['visit']
        row={'case_id':case_id,'patient':record['patient'],'split':record['split'],
             'source_sha256':original['source_sha256'],'legacy_cache_sha256':original['cache_sha256'],
             'legacy_image_sha256':array_sha(old_image),'support_sha256':array_sha(support),
             'valid_zero_voxels':int(np.count_nonzero(support & (old_image==0))),
             'outside_support_voxels':int(np.count_nonzero(~support)),
             'raw_zero_voxels':int(np.count_nonzero(raw_values==0)),
             'total_voxels':int(old_image.size),'cached_image_checked':False}
        model_image=old_image
        cache_path=legacy._safe(manifest_path.parent,original['cache_path'])
        if cache_path.is_file():
            cached=legacy.load_case(manifest_path,record)
            model_image=cached['image'];row['cached_image_checked']=True
            equal=np.array_equal(model_image,old_image)
            row['cached_image_matches_current_legacy_preprocessing']=bool(equal)
            if not equal:
                delta=np.abs(model_image.astype(np.float64)-old_image)
                row['cached_image_max_abs_difference']=float(delta.max())
                row['cached_image_different_voxels']=int(np.count_nonzero(delta))
                nonidentical_cache_inputs.append(case_id)
            row['cached_image_sha256']=array_sha(model_image)
            if np.any(model_image[~support]!=0):raise ValueError('Frozen cached image nonzero outside raw support')
        row['changed_mask_voxels']=int(np.count_nonzero((model_image==0)!=(~support)))
        if row['cached_image_checked'] and row['changed_mask_voxels']:
            changed=(model_image==0)!=(~support)
            row['changed_mask_gt_counts']=np.bincount(cached['mask'][changed],minlength=6).tolist()
        sidecar=out/'support'/f'{case_id}.npz'
        side_meta={'case_id':case_id,'split':record['split'],'support_policy':corrected.SUPPORT_POLICY,
                   'source_image_sha256':original['source_sha256']['image'],'cache_sha256':original['cache_sha256'],
                   'legacy_image_sha256':array_sha(model_image),'support_sha256':array_sha(support),
                   'affine_xyz':metadata['resampled_affine_xyz']}
        with sidecar.open('xb') as handle:np.savez_compressed(handle,support=support,metadata_json=np.asarray(json.dumps(side_meta,sort_keys=True)))
        row.update(support_path=str(sidecar.relative_to(out)),support_file_sha256=sha256_file(sidecar))
        records.append(row)
        group=groups[record['split']];group['cases']+=1
        for key in ('valid_zero_voxels','outside_support_voxels','changed_mask_voxels','raw_zero_voxels','total_voxels'):group[key]+=row[key]
        group['cached_images_checked']+=int(row['cached_image_checked'])
        if i%20==0 or i==211:print(f'Audited {i}/211; changed masks={sum(g["changed_mask_voxels"] for g in groups.values())}',flush=True)
    weights={}
    for model in ('unet2d','improved3d'):
        name=f'weights/{model}_best.pt';path=legacyroot/name
        if sha256_file(path)!=package['entries'][name]['sha256']:raise ValueError('Selected weight bytes changed')
        weights[model]=sha256_file(path)
    changed=sum(g['changed_mask_voxels'] for g in groups.values())
    public={'schema_version':1,'kind':'raw_support_equivalence_audit','status':'passed',
            'cases_total':211,'split_counts':{name:g['cases'] for name,g in groups.items()},
            'source_inventory_verified':True,'source_cache_images_checked':sum(g['cached_images_checked'] for g in groups.values()),
            'cache_manifest_sha256':sha256_file(manifest_path),'by_split':groups,
            'valid_zero_voxels_total':sum(g['valid_zero_voxels'] for g in groups.values()),
            'changed_mask_voxels_total':changed,'selected_weights_unchanged':True,
            'clinical_inference_performed':False,'legacy_source_sha256':expected,
            'support_policy':corrected.SUPPORT_POLICY,
            'metrics_provenance':'retained_after_zero_changed_masks' if not changed else 'affected_case_inference_required'}
    private={'created_at_utc':datetime.now(timezone.utc).isoformat(),'public_aggregate':public,
             'raw_root':str(raw),'legacy_source_root':str(legacyroot),'records':records,
             'preprocessed_images_byte_identical':True,'cached_inputs_different_from_current_cpu_preprocessing':nonidentical_cache_inputs,
             'selected_checkpoint_sha256':weights,'seconds':time.time()-started}
    save_json(out/'audit_private.json',private);save_json(out/'aggregate.json',public)
    print(json.dumps(public,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',required=True,type=Path);p.add_argument('--legacy-source-dir',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    run(p.parse_args())

if __name__=='__main__':main()
