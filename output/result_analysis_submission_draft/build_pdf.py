from pathlib import Path
import re,html,csv
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
base=Path(__file__).resolve().parent;root=base.parents[1]
fonts=next((root/'.venv').glob('lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf'))
for name,file in [('DV','DejaVuSans.ttf'),('DV-Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
styles={k:ParagraphStyle(k,fontName='DV-Bold' if k in ['title','heading'] else 'DV',fontSize=size,leading=lead,spaceAfter=space,textColor=colors.HexColor('#203044')) for k,size,lead,space in [('body',9,12.8,7),('cell',8,10.5,0),('heading',13,17,10),('title',21,26,14),('small',8,11,6)]}
def fmt(s):
 s=s.replace('−','-').replace('–','-').replace('—','-')
 s=re.sub(r'`([^`]+)`',r'\1',s)
 s=html.escape(s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 def link(m):
  label,url=m.groups()
  if url.startswith('http') and '127.0.0.1' not in url:return '<link href="'+url+'" color="#136c72">'+label+'</link>'
  return label
 return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
def p(s,sty='body'):return Paragraph(fmt(s),styles[sty])
lines=(base/'Result_Analysis.md').read_text().splitlines();story=[];i=0;width=522
while i<len(lines):
 s=lines[i].strip()
 if not s:i+=1;continue
 if s.startswith('# '):story+=[p('COSC 874 | Deep Learning II','small'),p(s[2:],'title')];i+=1;continue
 if s.startswith('## '):
  if re.match(r'## [2-5]\.',s):story.append(PageBreak())
  story.append(p(s[3:],'heading'));i+=1;continue
 if s.startswith('!['):
  # Main results are already tabulated. Include the more informative confusion figure.
  if 'confusion.png' in s:story+=[Image(str(base/'confusion.png'),width=width,height=width*810/2520),p('Confusion matrices: actual labels by row, predictions by column. S = SUPPORTS; R = REFUTES. Invalid outputs count as errors.','small')]
  i+=1;continue
 if s.startswith('[Full per-class TP'):
  counts=list(csv.DictReader((base/'per_class_counts.csv').open()))
  cells=[[p(x,'cell') for x in ['System / class','TP','FP','FN','TN']]]
  for row in counts:
   if row['System']=='Five-sentence RoBERTa':continue
   label=row['Class'].replace('NOT ENOUGH INFO','NEI')
   cells.append([p(row['System'].replace('RAG RoBERTa 256','RoBERTa').replace('RAG Llama','Llama')+' / '+label,'cell')]+[p(row[k],'cell') for k in ['TP','FP','FN','TN']])
  story.append(p('One-vs-rest counts: each class is positive; all other labels are negative.','small'))
  tab=Table(cells,colWidths=[242,70,70,70,70],repeatRows=1)
  tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f4')),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#d5dfe5'))]))
  story.append(tab);i+=1;continue
 if s.startswith('|'):
  data=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   raw=lines[i].strip();i+=1
   if re.fullmatch(r'[|\s:\-]+',raw):continue
   data.append([p(x.strip(),'cell') for x in raw.strip('|').split('|')])
  n=len(data[0]);widths=([260,87,87,88] if n==4 else [192,110,73,73,74] if n==5 else [160,362] if n==2 else [145,51,69,66,191] if n==5 else [width/n]*n)
  if n==5 and 'Claim' in data[0][0].getPlainText():widths=[118,65,68,68,203]
  table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('FONTNAME',(0,0),(-1,0),'DV-Bold'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#26727b')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe5'))]));story+=[table,Spacer(1,9)];continue
 buf=[s];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')):
  buf.append(lines[i].strip());i+=1
 story.append(p(' '.join(buf)))
out=root/'output/pdf/Result_Analysis_Final.pdf';out.parent.mkdir(exist_ok=True)
def footer(c,d):
 c.setFont('DV',8);c.setFillColor(colors.HexColor('#657484'));c.drawString(45,25,'COSC 874 | Result Analysis');c.drawRightString(567,25,str(d.page))
SimpleDocTemplate(str(out),pagesize=letter,leftMargin=45,rightMargin=45,topMargin=35,bottomMargin=45,title='FEVER Result Analysis',author='Praise Ben').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
