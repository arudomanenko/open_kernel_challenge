"""Check or update the CUDA source embedded in the standalone submission."""
import argparse
import ast
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Update the embedded CUDA source')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / 'submission.py'
    source = path.read_text()
    cuda = (root / 'kernels.cu').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_CUDA' for t in n.targets))
    if args.write:
        if '"""' in cuda:
            raise ValueError('CUDA source contains the Python string delimiter')
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = ['_CUDA = r"""\n' + cuda + '"""\n']
        updated = ''.join(lines)
        ast.parse(updated)
        path.write_text(updated)
        print('Updated embedded CUDA source; revalidate before submitting.')
    elif ast.literal_eval(node.value) != '\n' + cuda:
        raise SystemExit('Embedded source differs; run sync_cuda.py --write, then revalidate.')
    else:
        print('Embedded CUDA source matches kernels.cu.')


if __name__ == '__main__':
    main()
