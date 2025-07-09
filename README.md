```mermaid
graph TD
    subgraph Preprocessing
        A[DICOM Loader] --> B[dicom2nifti or SimpleITK]
        B --> C[PNG Conversion]
        C --> D[OCR Engine]
        D --> E[PDF Text Extractor]
    end

    subgraph ImageProcessing
        F[TotalSegmentator] --> G[Segmentation Masks]
        G --> H[Constraint Mapper]
    end

    subgraph NLP_and_Symbolic
        E --> I[Biomedical NLP]
        I --> J[NER + Parsing]
        J --> K[State Emitter]
        H --> K
        K --> L[Affinity Graph]
    end

    subgraph Inference
        L --> M[Constraint Lattice]
        M --> N[WildCore Security]
        N --> O[Phi-2 or Gemma-2B]
        O --> P[Symbolic Output]
    end

    subgraph ControlEvaluation
        P --> Q[State Vector & Metrics]
        Q --> R[Audit Logger]
        R --> S[Human-in-the-Loop]
    end
```

## Remote OHIF Access

The :mod:`msk_io.retrieval` package provides helpers for working with
web-based viewers. ``OHIFCanvasExtractor`` captures rendered frames using
a headless browser. ``DICOMStreamSniffer`` monitors network traffic to
download the original DICOM payloads. ``RemoteDICOMLoader`` combines both
methods and is triggered when ``remote_url`` and ``auth_token`` are set in
``PipelineSettings``.
