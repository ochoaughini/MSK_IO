from msk_io.schema.prompt_template import PromptTemplate, PromptParameter, PromptSet

DIAGNOSTIC_ASSESSMENT_PROMPT = PromptTemplate(
    template_name="DiagnosticAssessment",
    description="Generates an initial diagnostic assessment based on patient and image data.",
    template_string=(
        "You are an expert radiologist AI assistant. Analyze the following medical information "
        "and provide a concise diagnostic assessment. Focus on abnormalities, potential diagnoses, "
        "and their clinical significance.\n\n"
        "Patient Information:\n{patient_info_summary}\n\n"
        "Image Analysis Results:\n{image_analysis_summary}\n\n"
        "Relevant Clinical Context:\n{clinical_context}\n\n"
        "Your Assessment (structured JSON preferred):\n"
    ),
    parameters=[
        PromptParameter(name="patient_info_summary", description="Summary of patient demographics and history.", is_required=True),
        PromptParameter(name="image_analysis_summary", description="Summary of image segmentation and feature extraction results.", is_required=True),
        PromptParameter(name="clinical_context", description="Additional relevant clinical notes or lab results.", is_required=False, default_value="No additional context.")
    ],
    expected_output_format="JSON"
)

SEGMENTATION_FEEDBACK_PROMPT = PromptTemplate(
    template_name="SegmentationFeedback",
    description="Provides feedback on a segmentation result, identifying potential errors or areas for improvement.",
    template_string=(
        "You are an AI quality control assistant for medical image segmentation. "
        "Review the following segmentation results against the original image context. "
        "Identify any obvious errors, inconsistencies, or areas where segmentation could be improved.\n\n"
        "Original Image Context: {image_metadata_summary}\n"
        "Segmentation Results (ROIs and properties): {segmentation_details_json}\n\n"
        "Your Feedback (structured JSON preferred, e.g., {'errors': [], 'suggestions': []}):\n"
    ),
    parameters=[
        PromptParameter(name="image_metadata_summary", description="Summary metadata of the original image.", is_required=True),
        PromptParameter(name="segmentation_details_json", description="JSON representation of segmentation results.", is_required=True),
    ],
    expected_output_format="JSON"
)

REPORT_GENERATION_PROMPT = PromptTemplate(
    template_name="ReportGeneration",
    description="Generates a structured medical diagnostic report from analysis findings.",
    template_string=(
        "Synthesize the following diagnostic findings into a comprehensive medical report. "
        "The report should include patient information, study details, a summary of findings, "
        "and actionable recommendations. Maintain a professional, objective tone. "
        "Exclude disclaimers about AI generation unless explicitly requested.\n\n"
        "Patient Details: {patient_details_json}\n"
        "Study Details: {study_details_json}\n"
        "Diagnostic Findings: {findings_list_json}\n"
        "Image Analysis Summaries: {image_summaries_json}\n"
        "LLM Analysis Summaries: {llm_summaries_json}\n\n"
        "Your Medical Report (structured JSON preferred, conforming to DiagnosticReport schema):\n"
    ),
    parameters=[
        PromptParameter(name="patient_details_json", description="JSON of patient info.", is_required=True),
        PromptParameter(name="study_details_json", description="JSON of study info.", is_required=True),
        PromptParameter(name="findings_list_json", description="JSON list of DiagnosticFinding objects.", is_required=True),
        PromptParameter(name="image_summaries_json", description="JSON list of ImageAnalysisResult summaries.", is_required=True),
        PromptParameter(name="llm_summaries_json", description="JSON list of LLMAnalysisResult summaries.", is_required=True),
    ],
    expected_output_format="JSON"
)

LLM_AGENT_PROMPTS = PromptSet(
    set_name="LLMAgentPrompts",
    description="Collection of prompts used by various LLM agents in the MSK-IO pipeline.",
    prompts=[
        DIAGNOSTIC_ASSESSMENT_PROMPT,
        SEGMENTATION_FEEDBACK_PROMPT,
        REPORT_GENERATION_PROMPT
    ]
)

def get_prompt_template(template_name: str) -> PromptTemplate:
    for prompt in LLM_AGENT_PROMPTS.prompts:
        if prompt.template_name == template_name:
            return prompt
    raise ValueError(f"Prompt template '{template_name}' not found.")
