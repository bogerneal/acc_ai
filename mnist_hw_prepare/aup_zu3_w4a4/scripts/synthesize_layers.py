"""Resume stage F after installing the pinned vendor toolchain.

Gates on versions/part, synthesizes one small DWC first, then all other layers.
Hardware execution is untested until tools are available; this is not stitching.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from time import perf_counter
import xml.etree.ElementTree as ET

from build_hls_rtl import ROOT, sha, write_json


def run_command(argv, cwd, env, log, timeout):
    start = perf_counter()
    with log.open('w', encoding='utf-8') as output:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError(f'Timeout after {timeout}s: {argv}; see {log}')
    if code:
        raise RuntimeError(f'Exit {code}: {argv}; see {log}')
    return perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--timeout-sec', type=int, default=3600, help='Timeout per node')
    a = p.parse_args()
    if a.timeout_sec <= 0:
        p.error('timeout must be positive')
    out = a.run_dir.resolve()
    source = json.loads((out / 'report.json').read_text())
    assert source['status'] == 'sources_generated_hardware_validation_pending'
    report = dict(status='checking', layers=[], ipgen_seconds=None,
                  cppsim='not_run_by_this_script', stitched_core=False)
    logs = out / ('ipgen_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    logs.mkdir()
    try:
        tools = {n: shutil.which(n) for n in ('vivado', 'vitis_hls')}
        report['tools'] = tools
        missing = [n for n, path in tools.items() if path is None]
        if missing:
            report.update(status='blocked', reason='Missing WSL tools: ' + ', '.join(missing))
            return 2
        cfg = source['config']
        env = os.environ.copy()
        env.update(FINN_ROOT=source['dependency_snapshot']['runtime'], NUM_DEFAULT_WORKERS='1')
        for relative, expected in source['dependency_snapshot']['hashes'].items():
            assert sha(Path(env['FINN_ROOT']) / relative) == expected
        library = Path(env['FINN_ROOT']) / 'deps/finn-hlslib'
        from build_hls_rtl import command
        assert command(['git', '-C', str(library), 'rev-parse', 'HEAD']) == source['hlslib']['required_commit']
        assert not command(['git', '-C', str(library), 'status', '--porcelain'])
        report['versions'] = {}
        for name, tool in tools.items():
            log = logs / f'{name}_version.log'
            run_command([tool, '-version'], logs, env, log, 120)
            version = log.read_text(errors='replace')
            report['versions'][name] = version
            assert cfg['expected_vendor_version'] in version, 'Vendor version mismatch; review pinned FINN compatibility'
        report['part_check_seconds'] = run_command([tools['vivado'], '-mode', 'batch',
            '-source', str(ROOT / 'scripts/check_part.tcl'), '-tclargs', cfg['part']],
            logs, env, logs / 'part_check.log', 180)
        assert 'PART_CHECK_PASSED:' in (logs / 'part_check.log').read_text()
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.custom_op.registry import getCustomOp
        model = ModelWrapper(str(out / '08_codegen.onnx'))
        nodes = list(model.graph.node)
        nodes.sort(key=lambda n: not n.op_type.startswith('StreamingDataWidthConverter'))
        manifest = {row['node']: row for row in source['generated']}
        for index, node in enumerate(nodes):
            op = getCustomOp(node)
            directory = Path(op.get_nodeattr('code_gen_dir_ipgen'))
            for name, expected in manifest[node.name]['files'].items():
                assert sha(directory / name) == expected
            project = directory / f'project_{node.name}'
            assert not project.exists(), 'Use a fresh codegen run to avoid cached or partial synthesis'
            item = dict(node=node.name, is_toolchain_smoke_test=index == 0, status='running')
            report['layers'].append(item)
            start = perf_counter()
            try:
                argv = [tools['vitis_hls'], '-f', str(directory / f'hls_syn_{node.name}.tcl')]
                item['command'] = argv
                run_command(argv, directory, env, logs / f'{node.name}.log', a.timeout_sec)
                ip = project / 'sol1/impl/ip'
                rtl = project / 'sol1/impl/verilog'
                xml = project / f'sol1/syn/report/{node.name}_csynth.xml'
                assert (ip / 'component.xml').is_file() and list(rtl.glob('*.v')) and xml.is_file()
                op.set_nodeattr('ipgen_path', str(project))
                op.set_nodeattr('ip_path', str(ip))
                op.set_nodeattr('ip_vlnv', f'xilinx.com:hls:{node.name}:1.0')
                tree = ET.parse(xml)
                item['hls_estimates'] = {key: tree.findtext('.//' + key) for key in
                    ('EstimatedClockPeriod', 'Best-caseLatency', 'Worst-caseLatency',
                     'Interval-min', 'Interval-max', 'BRAM_18K', 'DSP', 'FF', 'LUT')}
                item.update(status='passed', xml_report=str(xml), rtl_directory=str(rtl))
            except Exception as exc:
                item.update(status='failed', error=str(exc))
                raise
            finally:
                item['seconds'] = perf_counter() - start
                write_json(logs / 'report.json', report)
            model.save(str(logs / 'partial_ipgen.onnx'))
        model.save(str(logs / '09_ipgen.onnx'))
        report.update(status='per_layer_ip_complete',
                      ipgen_seconds=sum(row['seconds'] for row in report['layers']))
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(logs / 'report.json', report)
        print(json.dumps(dict(report=str(logs / 'report.json'), status=report['status'])))


if __name__ == '__main__':
    raise SystemExit(main())
