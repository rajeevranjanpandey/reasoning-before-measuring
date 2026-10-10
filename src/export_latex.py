"""Export the current evidence-linked Markdown manuscript to editable XeLaTeX."""
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def escape(text):
    chars={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(chars.get(c,c) for c in text)

def prose(text):
    parts=re.split(r'(https?://\S+|\[\d+(?:,\s*\d+)*\])',text)
    out=[]
    for part in parts:
        if part.startswith(('https://','http://')):out.append(r'\url{'+part+'}')
        elif re.fullmatch(r'\[\d+(?:,\s*\d+)*\]',part):
            out.append(r'\cite{'+','.join('ref'+x for x in re.findall(r'\d+',part))+'}')
        else:out.append(escape(part).replace(r"\_", r"\_\allowbreak{}").replace(r'x\_\allowbreak{}S', r'$x_S$'))
    return ''.join(out)

def main():
    out=ROOT/'paper/latex';out.mkdir(exist_ok=True)
    source=ROOT/'paper/manuscript.md'
    if not source.exists():source=ROOT/'paper/pilot_manuscript.md'
    text=source.read_text()
    body,refs=text.split('## References',1)
    lines=body.splitlines();title=lines[0].removeprefix('# ')
    tex=[r'\documentclass[11pt,a4paper]{article}',r'\usepackage[margin=22mm]{geometry}',
         r'\usepackage{fontspec}',r'\setmainfont{lmroman10-regular.otf}[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf,BoldItalicFont=lmroman10-bolditalic.otf]',r'\usepackage{graphicx,longtable,booktabs,array,xurl,needspace}',
         r'\usepackage[hidelinks]{hyperref}',r'\setlength{\emergencystretch}{3em}',
         r'\title{'+escape(title)+'}',r'\author{}',r'\date{}',r'\begin{document}',r'\maketitle']
    i=1
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[- :]+',v) for v in row):table.append(row)
                i+=1
            n=len(table[0]);width=(.96/n)-.02
            tex += [r'{\small',r'\begin{longtable}{'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{width:.3f}'+r'\textwidth}' for _ in range(n))+'}',r'\toprule',
                    ' & '.join(prose(v) for v in table[0])+r' \\',r'\midrule',r'\endhead']
            tex += [' & '.join(prose(v) for v in row)+r' \\' for row in table[1:]]
            tex += [r'\bottomrule',r'\end{longtable}',r'}'];continue
        if line.startswith('Figure '):
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            following=re.fullmatch(r'!\[.*?\]\(([^)]+)\)',lines[j].strip()) if j<len(lines) else None
            if following:
                name=following[1];(out/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'paper'/name,out/name)
                tex += [r'\noindent\begin{minipage}{\linewidth}',prose(line),r'\begin{center}',r'\includegraphics[width=0.92\linewidth]{'+name+'}',r'\end{center}',r'\end{minipage}\par']
                i=j+1;continue
        if line.startswith('Table '):tex.append(r'\Needspace{12\baselineskip}')
        image=re.fullmatch(r'!\[.*?\]\(([^)]+)\)',line)
        if image:
            name=image[1];assert (ROOT/'paper'/name).exists();(out/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'paper'/name,out/name)
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            caption=lines[j].strip() if j<len(lines) and re.match(r'Figure \d+\.',lines[j].strip()) else None
            tex += [r'\noindent\begin{minipage}{\linewidth}',r'\begin{center}',r'\includegraphics[width=0.92\linewidth]{'+name+'}',r'\end{center}']
            if caption:tex.append(prose(caption));i=j
            tex += [r'\end{minipage}\par']
        elif line.startswith('### '):
            next_text=next((x.strip() for x in lines[i+1:] if x.strip()),'')
            if next_text.startswith('Table '):tex.append(r'\Needspace{16\baselineskip}')
            tex.append(r'\subsection{'+escape(re.sub(r'^\d+(?:\.\d+)*\s+','',line[4:]))+'}')
        elif line.startswith('## '):
            heading=re.sub(r'^\d+\s+','',line[3:])
            command=r'\section*' if not re.match(r'\d',line[3:]) else r'\section'
            tex.append(command+'{'+escape(heading)+'}')
        else:tex.append(prose(line)+'\n')
        i+=1
    bibliography=[r'\begin{thebibliography}{99}',r'\setlength{\itemsep}{2pt}']
    entries=re.findall(r'^\[(\d+)\]\s+(.+)$',refs,re.M)
    assert len(entries)==22
    for number,entry in entries:bibliography.append(r'\bibitem{ref'+number+'} '+prose(entry))
    bibliography.append(r'\end{thebibliography}')
    (out/'references.tex').write_text('\n'.join(bibliography)+'\n')
    tex += [r'\input{references.tex}',r'\end{document}']
    (out/'manuscript.tex').write_text('\n'.join(tex)+'\n')
    (out/'README.md').write_text('''# LaTeX source

This editable source reproduces the current manuscript text, the manuscript tables, figures and 22 references. Compile from this directory with `xelatex manuscript.tex` twice. A standard TeX Live installation with fontspec, graphicx, longtable, booktabs, array, xurl and hyperref is required. References are in references.tex; no BibTeX step is needed.

The accompanying manuscript.pdf was compiled with Tectonic 0.17.0 and visually reviewed. Rebuild with `tectonic --untrusted --keep-logs manuscript.tex` from this directory. The canonical ../manuscript.pdf was rendered from the Word version and has a different layout. This is generic article format, not certification of a journal-specific template. Build verification is recorded in ../../provenance/latex_build_verification.json.
''')
    print('Exported manuscript.tex, references.tex and the required figures')

if __name__=='__main__':main()
