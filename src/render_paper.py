"""Render DOCX with a configured document renderer or a local LibreOffice installation.

Set PAPER_DOCX_RENDERER to a compatible render_docx.py when available. Otherwise,
LibreOffice and pypdfium2 supply a portable fallback with the same core CLI.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import pypdfium2 as pdfium


def raster(pdf_path,dpi=150,output_folder=None,**kwargs):
    folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
    pdf=pdfium.PdfDocument(pdf_path);paths=[]
    try:
        for index in range(len(pdf)):
            page=pdf[index];bitmap=page.render(scale=dpi/72)
            path=folder/f'page-{index+1}.png';bitmap.to_pil().save(path);paths.append(str(path))
            bitmap.close();page.close()
    finally:pdf.close()
    return paths


def main():
    configured=os.environ.get('PAPER_DOCX_RENDERER')
    candidates=[Path(configured)] if configured else sorted((Path.home()/'.codex/plugins/cache/openai-primary-runtime/documents').glob('*/skills/documents/render_docx.py'),reverse=True)
    for script in candidates:
        if script.is_file():
            spec=importlib.util.spec_from_file_location('packaged_renderer',script)
            renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
            renderer.convert_from_path=raster
            return renderer.main()
    if configured:raise FileNotFoundError('PAPER_DOCX_RENDERER does not point to a file')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--emit_pdf',action='store_true');parser.add_argument('--dpi',type=int,default=150)
    args=parser.parse_args()
    executable=shutil.which('libreoffice') or shutil.which('soffice')
    mac=Path('/Applications/LibreOffice.app/Contents/MacOS/soffice')
    if not executable and mac.exists():executable=str(mac)
    if not executable:raise RuntimeError('Install LibreOffice or set PAPER_DOCX_RENDERER to a compatible render_docx.py. See REPRODUCIBILITY.md.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='paper-render-profile-') as profile:
        subprocess.run([executable,'-env:UserInstallation='+Path(profile).as_uri(),'--headless','--convert-to','pdf','--outdir',str(args.output_dir.resolve()),str(args.input.resolve())],check=True,timeout=180)
    pdf=args.output_dir/(args.input.stem+'.pdf')
    if not pdf.exists():raise RuntimeError('LibreOffice did not produce the expected PDF')
    raster(pdf,dpi=args.dpi,output_folder=args.output_dir)
    if not args.emit_pdf:pdf.unlink()


if __name__=='__main__':main()
