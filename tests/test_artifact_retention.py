import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArtifactRetentionTests(unittest.TestCase):
    def test_concurrent_history_appends_keep_both_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "history.csv")
            code = '''
import sys
import pandas as pd
from nba_props import _append_only_history
for index in range(15):
    _append_only_history(sys.argv[1], pd.DataFrame([{"id": sys.argv[2]+str(index)}]), ["id"])
'''
            processes = []
            try:
                for prefix in ["A", "B"]:
                    processes.append(subprocess.Popen(
                        [sys.executable, "-c", code, target, prefix], cwd=ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    ))
                for process in processes:
                    _, error = process.communicate(timeout=60)
                    self.assertEqual(process.returncode, 0, error)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                for process in processes:
                    process.communicate()
            import pandas as pd
            frame = pd.read_csv(target)
            self.assertEqual(len(frame), 30)
            self.assertEqual(frame.id.nunique(), 30)

    def run_isolated(self, script):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", textwrap.dedent(script), directory],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_direct_replace_append_delete_and_restore_bytes(self):
        result = self.run_isolated('''
            import sys,json,gzip,hashlib,os
            from pathlib import Path
            from artifact_retention import install_retention
            root=Path(sys.argv[1]); (root/'data').mkdir()
            path=root/'data/lines.csv'; path.write_bytes(b'first')
            store=install_retention(root)
            path.write_bytes(b'second')
            staged=root/'data/new.tmp'; staged.write_bytes(b'third')
            os.replace(staged,path)
            with path.open('ab') as f:f.write(b'-appended')
            path.unlink()
            values={gzip.decompress(p.read_bytes()).decode() for p in (root/'.retained_versions/objects').glob('*.gz')}
            for p in (root/'.retained_versions/objects').glob('*.gz'):
                assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==p.stem
            records=[json.loads(p.read_text()) for p in (root/'.retained_versions/records').rglob('*.json')]
            assert all(r['path']=='data/lines.csv' for r in records)
            print(json.dumps(sorted(values)))
        ''')
        self.assertEqual(result, ["first", "second", "third", "third-appended"])

    def test_path_truncate_and_byte_path_replace_are_preserved(self):
        result = self.run_isolated('''
            import sys,json,gzip,os
            from pathlib import Path
            from artifact_retention import install_retention
            root=Path(sys.argv[1]); (root/'data').mkdir()
            path=root/'data/history.csv'; path.write_bytes(b'rows')
            store=install_retention(root)
            os.truncate(str(path), 0)
            staged=root/'data/new.tmp'; staged.write_bytes(b'replacement')
            os.replace(os.fsencode(str(staged)), os.fsencode(str(path)))
            values={gzip.decompress(p.read_bytes()).decode() for p in (root/'.retained_versions/objects').glob('*.gz')}
            print(json.dumps(sorted(values)))
        ''')
        self.assertIn("rows", result)
        self.assertIn("replacement", result)

    def test_retention_failure_blocks_overwrite(self):
        result = self.run_isolated('''
            import sys,json
            from pathlib import Path
            from artifact_retention import install_retention
            root=Path(sys.argv[1]); (root/'output').mkdir()
            path=root/'output/picks_latest.csv';path.write_text('keep')
            store=install_retention(root)
            original=store.preserve
            def fail(*a,**k):raise OSError('simulated full disk')
            store.preserve=fail
            try:path.write_text('lost')
            except OSError:pass
            store.preserve=original
            print(json.dumps(path.read_text()))
        ''')
        self.assertEqual(result, "keep")

    def test_duplicate_contents_share_one_object_and_checkpoint_keeps_new_files(self):
        result = self.run_isolated('''
            import sys,json
            from pathlib import Path
            from artifact_retention import install_retention
            root=Path(sys.argv[1]); (root/'output').mkdir()
            store=install_retention(root)
            for name in ['picks_today.csv','picks_archive.csv']:
                (root/'output'/name).write_text('same data')
            manifest,count=store.checkpoint(all_files=True)
            assert count==2
            saved=json.loads(manifest.read_text())
            print(json.dumps([len(list((root/'.retained_versions/objects').glob('*.gz'))),len(saved['versions'])]))
        ''')
        self.assertEqual(result, [1, 2])


if __name__ == "__main__":
    unittest.main()
