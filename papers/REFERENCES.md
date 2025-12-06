# Papers & References

This directory contains the papers that motivated and informed the design of this codebase.

## Papers

### Paper 1: Parameter Superposition (PSP)
- **Citation / arXiv**: Cheung et al., "Parameter Superposition", arXiv:1902.05522 (2019)
- **Authors**: Cheung, Chai, et al.
- **Year**: 2019
- **Status**: ✅ Implemented
- **Summary**: Introduces the idea of storing multiple task-specific models within a single parameter tensor by applying task-specific context vectors (e.g., binary masks or complex rotations) to the shared parameters. Enables compact multi-task storage and efficient task retrieval via context keys. Demonstrated on image-classification continual-learning benchmarks (e.g., permuted MNIST).
- **Relevant Code**:
  - `src/hands_on_neuroai/models/psp.py` - PSP-style parameter superposition layers
  - `src/hands_on_neuroai/models/context.py` - Context generation utilities (binary/complex/rotation)
  - `src/hands_on_neuroai/training/continual_learning.py` - General training loop used in experiments
- **Notebooks / Experiments**:
  - `notebooks/02_analysis.ipynb` - MNIST-specific PSP analysis (baseline vs PSP)
  - `notebooks/04_analysis_continual_learning.ipynb` - Dataset-agnostic comparison
- **Key Experiments Reproduced**:
  - Permuted MNIST tasks with binary/complex/rotation contexts
  - Hidden representation analysis via PCA
  - Task 0 accuracy retention during continual learning
- **PDF**: `cheung_et_al_2019.pdf`

#### BibTeX (arXiv placeholder)
```
@article{cheung2019parameter,
  title={Parameter Superposition},
  author={Cheung, et al.},
  journal={arXiv preprint arXiv:1902.05522},
  year={2019}
}
```

### Paper 2: Category-orthogonal object features guide information processing in recurrent neural networks trained for object categorization
- **Citation / arXiv**: Thorat, Aldegheri, Kietzmann, arXiv:2111.07898
- **Authors**: Sushrut Thorat, Giacomo Aldegheri, Tim C. Kietzmann
- **Year**: 2021
- **Status**: 📋 Planned for Implementation
- **Summary**: [Add your own 1–2 sentence summary of the paper here once you decide how it connects to this codebase.]
- **Planned Code Mapping**: [To be defined once experiments are implemented.]
- **PDF**: `thorat_et_al_2022.pdf`

#### BibTeX (to be filled)
```bibtex
@article{thorat2021categoryorthogonal,
  title={Category-orthogonal object features guide information processing in recurrent neural networks trained for object categorization},
  author={Thorat, Sushrut and Aldegheri, Giacomo and Kietzmann, Tim C.},
  journal={arXiv preprint arXiv:2111.07898},
  year={2021}
}
```

---

## Implementation Plan

1. **Phase 1** (✅ Complete): PSP experiments and visualization
   - Reproduce permuted MNIST results
   - Compare baseline vs context-aware (PSP) models
   - Analyze learned representations via PCA

2. **Phase 2** (🔄 Planned): Superposition / model-composition experiments
   - Implement the superposition architecture described by Thorat et al.
   - Add tests/notebooks that reproduce the paper's key experiments
   - Compare against the PSP implementation and baseline methods

---

## How to add / update entries

1. Drop the PDF into this `papers/` directory.
2. Update the corresponding entry above with full author list, DOI/arXiv URL, and a complete BibTeX entry.
3. Link notebooks and code files that implement the experiments.


