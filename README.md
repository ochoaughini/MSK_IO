```mermaid
graph TD
    subgraph Preprocessing
        A[DICOM Loader] --> B[dicom2nifti / SimpleITK]
        B --> C[PNG Conversion]
        C --> D[OCR Engine (Tesseract/EasyOCR)]
        D --> E[PDF Text Extractor (pdf2image + PyMuPDF)]
    end

    subgraph Image Processing
        F[TotalSegmentator] --> G[Segmentation Masks]
        G --> H[Anatomical Constraint Mapper]
    end

    subgraph NLP & Symbolic Analysis
        E --> I[scispaCy / medspaCy]
        I --> J[Named Entity Recognition]
        J --> K[SymbolicStateEmitter]
        H --> K
        K --> L[SymbioticAffinityGraph]
    end

    subgraph Inference Engine
        L --> M[Constraint Lattice Core]
        M --> N[WildCore Security Layer]
        N --> O[Phi-2 or Gemma-2B via Ollama / transformers]
        O --> P[Symbolic Reasoning Output]
    end

    subgraph Control & Evaluation
        P --> Q[State Vector + EmancipationMetric]
        Q --> R[Audit Logger / Memory Tracker]
        R --> S[Result Validator (Human-in-the-Loop)]
    end

    style A fill:#f9f,stroke:#333,stroke-width:1px
    style O fill:#bbf,stroke:#000,stroke-width:1px
```
