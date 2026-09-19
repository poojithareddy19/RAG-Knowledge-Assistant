from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

output_path = Path(__file__).with_name('project_brief_a_to_g.pdf')

styles = getSampleStyleSheet()
styles['BodyText'].fontName = 'Helvetica'
styles['BodyText'].fontSize = 10
styles['BodyText'].leading = 14

heading_style = ParagraphStyle(
    name='Heading',
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    spaceBefore=10,
    spaceAfter=6,
)
title_style = ParagraphStyle(
    name='Title',
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=28,
    textColor=colors.HexColor('#0F4C81'),
    spaceAfter=12,
)
subtitle_style = ParagraphStyle(
    name='Subtitle',
    fontName='Helvetica',
    fontSize=12,
    textColor=colors.grey,
    spaceAfter=18,
)

content = [
    Paragraph('Production RAG Assistant', title_style),
    Paragraph('Brief Overview from A to G', subtitle_style),
    Paragraph('<b>A. What this project is</b>', heading_style),
    Paragraph('This project is a production-style Retrieval-Augmented Generation (RAG) assistant for answering questions from your own documents. It is designed to provide grounded answers with citations, confidence scoring, and safe fallback behavior.'),
    Spacer(1, 8),
    Paragraph('<b>B. The problem it solves</b>', heading_style),
    Paragraph('Organizations often keep critical knowledge in policies, manuals, contracts, and SOPs. This assistant helps employees find accurate answers without relying on guesswork or unverified model output.'),
    Spacer(1, 8),
    Paragraph('<b>C. How it works</b>', heading_style),
    ListFlowable([
        ListItem(Paragraph('Documents are ingested and split into searchable chunks.')),
        ListItem(Paragraph('The user question is embedded and matched against the indexed chunks.')),
        ListItem(Paragraph('A confidence gate checks whether the retrieved evidence is strong enough.')),
        ListItem(Paragraph('If the evidence is sufficient, the system generates a grounded answer with citations.')),
        ListItem(Paragraph('If not, it declines to answer rather than hallucinating.')),
    ], bulletType='bullet', bulletFontName='Helvetica', bulletFontSize=10, start='•', spaceAfter=8),
    Spacer(1, 8),
    Paragraph('<b>D. Main features</b>', heading_style),
    ListFlowable([
        ListItem(Paragraph('Cited answers with document, page, chunk, and passage references.')),
        ListItem(Paragraph('Confidence estimation based on retrieval signals.')),
        ListItem(Paragraph('Two hallucination guards: retrieval threshold and generation fallback.')),
        ListItem(Paragraph('Structured logging and monitoring for interactions.')),
        ListItem(Paragraph('Evaluation support with retrieval metrics and a gold dataset.')),
    ], bulletType='bullet', bulletFontName='Helvetica', bulletFontSize=10, start='•', spaceAfter=8),
    Spacer(1, 8),
    Paragraph('<b>E. Technology stack</b>', heading_style),
    Paragraph('The system is built with Python, Streamlit for the UI, FAISS for vector search, sentence-transformers for embeddings, and Ollama running local open-weight models as the generation backend.'),
    Spacer(1, 8),
    Paragraph('<b>F. How to run it</b>', heading_style),
    Paragraph('Install the dependencies from the requirements file, configure the model provider in the environment settings, and launch the Streamlit app to upload documents and ask questions.'),
    Spacer(1, 8),
    Paragraph('<b>G. Why it matters</b>', heading_style),
    Paragraph('This project demonstrates a practical and trustworthy approach to enterprise knowledge search. It keeps knowledge external, makes answers attributable, and helps teams adopt AI responsibly.'),
]

doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
doc.build(content)
print(f'Created {output_path}')
