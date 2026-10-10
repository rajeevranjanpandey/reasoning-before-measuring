"""Draw study schematics from the documented protocol; no outcome data are encoded."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
INK='#182430';BLUE='#dceaf3';GRAY='#f1f3f4';TEAL='#dfefea'
def box(ax,x,y,w,h,title,body,fill=GRAY):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.05',linewidth=1,edgecolor=INK,facecolor=fill))
 ax.text(x+w/2,y+h-.19,title,ha='center',va='top',weight='bold',fontsize=10,color=INK)
 ax.text(x+w/2,y+h-.52,body,ha='center',va='top',fontsize=9.4,color=INK,linespacing=1.35)
def arrow(ax,start,end,style='-'):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,lw=1.2,linestyle=style,color=INK))
def save(fig,name):
 for ext in ['svg','png','pdf']:fig.savefig(ROOT/'paper'/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def main(full_study=False):
 fig,ax=plt.subplots(figsize=(9,4.3));ax.set(xlim=(-.12,9.0),ylim=(-.15,4.3));ax.axis('off')
 box(ax,.05,1.78,1.9,1.30,'Observed state','Revealed values\nLegal sensor IDs\nRemaining budget',BLUE)
 box(ax,2.35,1.78,1.9,1.30,'Acquisition policy','Classical rule OR\nfixed model plan OR\nrepeated model',BLUE)
 box(ax,4.65,1.78,1.9,1.30,'Reveal one value','Request a new ID\nReturn its recorded\nmeasurement',TEAL)
 box(ax,6.95,1.78,1.9,1.30,'Shared predictor','Update probabilities\nusing the same\nmasked classifier',TEAL)
 for a,b in [(1.95,2.35),(4.25,4.65),(6.55,6.95)]:arrow(ax,(a,2.43),(b,2.43))
 ax.text(5.6,4.04,'Hidden snapshot archive',ha='center',weight='bold',fontsize=10)
 ax.text(5.6,3.73,'41 recorded channels, disclosed only on request',ha='center',fontsize=9)
 arrow(ax,(5.6,3.48),(5.6,3.10),'--')
 ax.plot([7.9,7.9,1.0,1.0],[1.76,1.21,1.21,1.61],color=INK,lw=1.2)
 arrow(ax,(1.0,1.50),(1.0,1.78))
 ax.text(4.45,1.00,'Repeat while fewer than B distinct readings have been acquired',ha='center',fontsize=9.7)
 ax.text(.05,.32,'At B readings: final prediction → paired diagnostic evaluation',ha='left',weight='bold',fontsize=10)
 ax.text(.05,-.03,'Primary model budget B = 3. Ground-truth labels are used only for evaluation.',ha='left',fontsize=9.3)
 save(fig,'acquisition_workflow')
 fig,ax=plt.subplots(figsize=(9,4.8));ax.set(xlim=(-.1,9.0),ylim=(-.2,4.8));ax.axis('off')
 ax.text(0,4.5,'A  Publisher training partition',weight='bold',fontsize=11)
 for x,title,body in [(0,'Fit predictor','Runs 1–350\n7,350 snapshots'),(3.1,'Select on validation','Runs 351–400\n1,050 snapshots'),(6.2,'Develop interface','Runs 401–420\nDevelopment pool')]:box(ax,x,2.85,2.65,1.25,title,body,BLUE)
 ax.text(0,2.35,'B  Publisher testing partition',weight='bold',fontsize=11)
 for x,title,body,fill in [(0,'Reported pilot','Runs 1–20\n420 snapshots',TEAL),(3.1,'Separate confirmation',('Runs 21–70\n1,050 snapshots' if full_study else 'Runs 21–70\nNot reported as results'),GRAY),(6.2,'Predictor extension',('Runs 71–90\n420 snapshots' if full_study else 'Runs 71–90\nNot reported as results'),GRAY)]:box(ax,x,.65,2.65,1.25,title,body,fill)
 ax.text(0,.23,'One snapshot per run and class. Identities grouped across all 21 operating classes.',fontsize=9.4)
 ax.text(0,-.11,'Testing runs 91–500 remain reserved. Boxes indicate roles, not proportional sample sizes.',fontsize=9.4)
 save(fig,'study_design_full' if full_study else 'study_design')
 print('Two protocol diagrams generated as editable SVG and 300 dpi PNG')
if __name__=='__main__':
 import sys
 main('--full-study' in sys.argv)
