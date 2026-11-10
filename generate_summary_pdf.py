import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from pypdf import PdfReader

def generate_pdf(filename="reactive_optimizers_summary.pdf"):
    # Target: Exactly 1 page (Letter: 612 x 792 pt, margins 36 pt = 0.5 in)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1E293B'),
        alignment=0,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=6
    )
    
    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12.5,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=5,
        spaceAfter=2.5
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.1,
        leading=10.6,
        textColor=colors.HexColor('#334155'),
        alignment=4, # Justified
        spaceAfter=3
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=10.4,
        textColor=colors.HexColor('#334155'),
        leftIndent=10,
        spaceAfter=2
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.0,
        textColor=colors.HexColor('#1E3A8A'),
    )

    story = []
    
    # Header Title
    story.append(Paragraph("Reactive Optimizers: Architecture, Limitations, and Opportunities in Federated Learning", title_style))
    story.append(Paragraph("<b>Reading Summary & Architectural Analysis</b> | Reference: Bai et al. (MIT, 2024 / arXiv:2610.33826) | Screening Task Submission", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=5))
    
    # Section 1: What It Does
    story.append(Paragraph("1. Core Concept & Mechanism: What OPAL Does", sec_heading))
    p1 = (
        "Standard deep neural network training relies on <i>static optimization recipes</i>—pre-configured update rules "
        "(e.g., AdamW, SGD) with predefined learning rate schedules that remain rigid throughout training regardless of unexpected "
        "training hurdles. In contrast, <i>learned optimizers</i> (trained via meta-learning or RL) act as opaque black-boxes "
        "that suffer from extreme computational cost and catastrophic generalization failure outside their training distribution. "
        "<b>OPAL (Optimizer with an Agent in the Loop)</b> introduces a paradigm shift: placing an autonomous LLM coding agent "
        "(e.g., Claude Code) directly into the operational training loop to dynamically inspect training telemetry and rewrite executable "
        "optimizer source code in real time."
    )
    story.append(Paragraph(p1, body_style))
    
    p2 = (
        "OPAL decouples the optimization loop into two co-evolving, agent-modifiable executable programs: "
        "<b>(1) The Analysis Program:</b> A custom Python telemetry script authored by the agent to compute real-time training diagnostics "
        "(e.g., layer-wise gradient norms, Hessian curvature approximations, loss stall/plateau detectors, and activation distributions). "
        "<b>(2) The Optimizer Program:</b> Executable PyTorch logic executing the gradient-to-parameter update mapping. "
        "At periodic checkpoints (every <i>K</i> steps), the agent evaluates the telemetry emitted by the analysis program and dynamically refactors "
        "the optimizer code—synthesizing adaptive learning rate warmups/decays, custom layer-specific momentum dampening, numerical precision changes, "
        "and spike-recovery guards. This enables zero-shot optimization adaptation without task-specific meta-training."
    )
    story.append(Paragraph(p2, body_style))
    
    # Section 2: Limitation
    story.append(Paragraph("2. Critical Limitation: Latency Bottleneck and Code Execution Safety", sec_heading))
    p3 = (
        "A fundamental limitation of OPAL lies in the <b>substantial latency overhead, monetary cost, and reliability risks</b> "
        "incurred by synchronous LLM agent invocations within the training loop. While standard PyTorch optimizer steps take mere milliseconds "
        "on modern accelerators (GPUs/TPUs), invoking a foundation model agent to parse telemetry, generate code, and execute modifications "
        "requires multiple seconds per reflection event. In high-throughput, latency-critical, or large-scale distributed training regimes, "
        "frequent agent interventions introduce prohibitive idle overhead. Furthermore, because LLM code generation is inherently stochastic "
        "and unverified by formal methods, there is a non-trivial risk of generating code with subtle numerical instabilities (e.g., vanishing/exploding "
        "gradients, division by near-zero variance, GPU memory leaks from uncleaned computational graphs), or catastrophic divergence if the agent misinterprets "
        "transient loss noise as systematic divergence."
    )
    story.append(Paragraph(p3, body_style))
    
    # Section 3: Idea for Federated Learning
    story.append(Paragraph("3. Application to Federated Learning: Dynamic Client Heterogeneity & Noise Adaptation", sec_heading))
    p4 = (
        "Federated Learning (FL) fundamentally struggles with two intertwined challenges: <b>statistical heterogeneity (non-IID data skew)</b> "
        "and <b>corrupted/noisy client participation</b> (e.g., label noise). Static algorithms like FedAvg suffer from severe client drift, "
        "while FedProx mitigates drift through a proximal regularizer &mu;/2 ||w - w<sup>t</sup>||<sup>2</sup> with a fixed, hand-tuned hyperparameter &mu;. "
        "A Reactive Optimizer agent situated on the central FL server could revolutionize adaptive robust aggregation:"
    )
    story.append(Paragraph(p4, body_style))
    
    story.append(Paragraph(
        "&bull; <b>Telemetry-Driven Dynamic Proximal Scaling:</b> The server agent tracks per-client telemetry across communication rounds—specifically "
        "local training loss trajectories, client update norms ||&Delta;w<sub>i</sub><sup>t</sup>||, and directional cosine alignment relative to the global consensus "
        "cos(&Delta;w<sub>i</sub><sup>t</sup>, &Delta;w<sub>global</sub><sup>t</sup>). Rather than fixing &mu;, the agent dynamically rewrites the client optimization objective, "
        "synthesizing client-specific proximal penalties &mu;<sub>i</sub>(t) that increase when client drift is acute and relax when alignment is strong.",
        bullet_style
    ))
    story.append(Paragraph(
        "&bull; <b>Reactive Byzantine / Label-Noise Mitigation:</b> When a client exhibits severe label corruption (such as Client 4 with 40% label noise in PathMNIST), "
        "its updates produce erratic update norms and orthogonal/negative cosine similarity against clean clients. Instead of relying on rigid, pre-defined robust "
        "aggregation rules (like Trimmed Mean or Coordinate-wise Median, which fail under heavy Dirichlet &alpha;=0.3 skew), the reactive server agent can dynamically "
        "diagnose anomalous drift and synthesize custom projection, downweighting, or noise-filtering aggregation programs on the fly.",
        bullet_style
    ))
    
    # Mini summary box at bottom
    summary_box_data = [[
        Paragraph(
            "<b>Key Takeaway for FL:</b> Reactive optimization transforms federated aggregation from a static heuristic into an active, "
            "context-aware closed-loop control system capable of autonomously isolating label-corrupted nodes and balancing client drift in real time.",
            callout_style
        )
    ]]
    summary_table = Table(summary_box_data, colWidths=[540])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#93C5FD')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(Spacer(1, 4))
    story.append(summary_table)

    doc.build(story)
    
    reader = PdfReader(filename)
    num_pages = len(reader.pages)
    print(f"Generated {filename} with {num_pages} page(s).")
    return num_pages

if __name__ == "__main__":
    generate_pdf()

