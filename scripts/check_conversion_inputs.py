#!/usr/bin/env python3
"""Run public shell CLI against read-only real-data fixtures, including intentional rejections."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--excerpt',type=Path,default=Path('logs/conversion_simplification/inputs/bimanual_excerpt.mcap'))
    args=parser.parse_args()
    import uuid
    run_id=uuid.uuid4().hex[:8]
    cases=[
        ('single_raw','single','/source/single_raw',True),
        ('single_mcap','single','/source/single/20260825_224317/20260825_224317.mcap',True),
        ('bimanual_raw','bimanual','/source/bimanual/20261015_164839',True),
        ('bimanual_mcap_excerpt','bimanual',str(args.excerpt),True),
        ('single_irregular_raw','single','/source/single/20260825_224317',True),
        ('bimanual_irregular_raw','bimanual','/source/bimanual/20261015_165042',True),
        ('bimanual_stale_camera','bimanual','/source/bimanual/20261015_164839/episode_1792050464343925529.mcap',False),
    ]
    report=[]
    for name,robot,source,success in cases:
        command=['sh','scripts/convert.sh','--robot',robot,'--dataset','conversion_check_'+name,
                 '--input',source,'--output',str(args.directory/'cli-outputs'/run_id/name),'--task','Test recorded data.','--validate-only']
        result=subprocess.run(command,capture_output=True,text=True)
        assert (result.returncode==0)==success,(name,result.stdout,result.stderr)
        report.append(dict(case=name,expected_success=success,returncode=result.returncode,error=result.stderr.strip()))
    (args.directory/'cli-input-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
