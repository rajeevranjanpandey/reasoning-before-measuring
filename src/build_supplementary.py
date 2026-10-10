"""Assemble audited numeric tables and six descriptive class figures for review."""
import re, io
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
ROOT=Path(__file__).resolve().parents[1]

def main():
    styles=getSampleStyleSheet()
    styles['Normal'].fontName='Times-Roman'; styles['Normal'].fontSize=10; styles['Normal'].leading=12
    styles['Heading1'].fontName='Times-Bold';styles['Heading1'].fontSize=13
    styles['Heading2'].fontName='Times-Bold';styles['Heading2'].fontSize=11
    styles['Title'].fontName='Times-Bold';styles['Title'].fontSize=16
    cell=styles['Normal'].clone('cell');cell.fontSize=8;cell.leading=10
    source=(ROOT/'paper/postreview/manuscript_sections.md').read_text()
    source=source.replace('# Results from the completed post-review studies','# Supplementary numerical results')
    source=source.replace('This evidence-linked draft reports all planned follow-up studies. It is separate from the original pilot and requires integration into the abstract, results, discussion and conclusion, followed by scientific and visual review.', 'This supplement reports complete audited follow-up results accompanying the manuscript. The pilot results and primary findings are presented in the main paper.')
    lines=source.splitlines(); story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[- :]+',v) for v in row):rows.append(row)
                i+=1
            n=len(rows[0]);weights=[max(12,min(26,max(len(r[c]) for r in rows))) for c in range(n)]
            widths=[499*w/sum(weights) for w in weights]
            table=LongTable([[Paragraph(escape(v),cell) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.lightgrey),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
            story.extend([table,Spacer(1,10)]);continue
        style=styles['Normal']
        for prefix,name in [('### ','Heading2'),('## ','Heading1'),('# ','Title')]:
            if line.startswith(prefix):line=line[len(prefix):];style=styles[name];break
        story.extend([Paragraph(escape(line),style),Spacer(1,7)]);i+=1
    out=ROOT/'paper/supplementary';out.mkdir(exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Times-Roman',9);canvas.drawCentredString(A4[0]/2,22,str(doc.page))
    SimpleDocTemplate(str(out/'numerical_results.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=38,author='',title='Supplementary numerical results').build(story,onFirstPage=footer,onLaterPages=footer)
    figstory=[]
    captions=[('closed_loop_class_recall.png','Supplementary Figure S5. Class-specific recall with the selected predictor on testing runs 71–90. Each class has twenty identities. Model recalls average five decoding seeds. All policies use three sensors except the full reference with 41 sensors. Colors share a 0–100% scale. Thin row and column dividers separate policies and classes. These are descriptive estimates without classwise tests.'),('closed_loop_class_spider.png','Supplementary Figure S6. Spider comparison of the same selected-predictor class recalls. Both panels use the same forward and rounded-tool references. Every spoke has a linear 0–100% scale in dataset class order. Connecting lines aid visual comparison, without area scoring, class-adjacency interpretation or classwise confidence intervals.')]
    for k,(name,caption) in enumerate(captions):
        if k:figstory.append(PageBreak())
        figstory.append(Paragraph(caption,styles['Normal']));figstory.append(Spacer(1,16))
        img=Image(str(ROOT/'paper/postreview'/name));ratio=img.imageHeight/img.imageWidth;img.drawWidth=499;img.drawHeight=499*ratio;figstory.append(img)
        figstory.append(Spacer(1,16));figstory.append(Paragraph('Exact values accompany closed_loop_class_recall.csv. Visualization code was prepared with ChatGPT/Codex from recorded audited results.',styles['Normal']))
    SimpleDocTemplate(str(out/'selected_predictor_figures.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=38,author='',title='Selected-predictor class diagnostics').build(figstory,onFirstPage=footer,onLaterPages=footer)
    writer=PdfWriter()
    for name in ('numerical_results.pdf','confusion_matrices.pdf','selected_predictor_figures.pdf'):
        for page in PdfReader(out/name).pages:
            old=[]
            def find_footer(text,cm,tm,font,size):
                if text.strip().isdigit() and 0 <= tm[5] < 55:old.append((tm[4],tm[5],size))
            page.extract_text(visitor_text=find_footer)
            overlay=io.BytesIO(); c=canvas.Canvas(overlay,pagesize=(float(page.mediabox.width),float(page.mediabox.height)))
            c.setFillColor(colors.white)
            if name == 'confusion_matrices.pdf':
                c.rect(float(page.mediabox.width)/2-15, 34, 30, 15, fill=1, stroke=0)
            for x,y,size in old:c.rect(x-4,y-4,24,size+8,fill=1,stroke=0)
            c.setFillColor(colors.black);c.setFont('Times-Roman',9)
            c.drawCentredString(float(page.mediabox.width)/2,22,str(len(writer.pages)+1));c.save()
            page.merge_page(PdfReader(overlay).pages[0]);writer.add_page(page)
    writer.add_metadata({'/Title':'Supplementary material: Reasoning Before Measuring','/Author':''})
    with (out/'supplementary_material.pdf').open('wb') as f:writer.write(f)
    print('Supplementary numerical tables and Figures S1–S6 assembled')
if __name__=='__main__':main()
