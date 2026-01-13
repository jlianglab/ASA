# ASA: Learning Anatomical Consistency, Sub-volume Spatial Relationships and Fine-grained Appearance for CT Images
This repository provides a PyTorch implementation of the ASA: Learning Anatomical Consistency, Sub-volume Spatial Relationships and Fine-grained Appearance for CT Images.

We develop ASA to learn anatomical consistency, sub-volume spatial relationships, and fine-grained appearance. ASA incorporates four learning perspectives: (1) capturing sub-volume relationships through 3D sub-volume order prediction, (2) depicting fine-grained features within volumes through volume appearance recovery, (3) comprehending high-level global features by maximizing the agreement between two spatially related views using the student-teacher network, and (4) acquiring local features at the sub-volume level by aligning the shared local views within two spatially related views.
<p align="center"><img width="100%" src="images/fig_main.png" /></p>

## Publication
<b>ASA: Learning Anatomical Consistency, Sub-volume Spatial Relationships and Fine-grained Appearance for CT Images </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [DongAo Ma](https://github.com/Mda233)<sup>1</sup>, [Ziyu Zhou](https://scholar.google.com/citations?user=nvAfKnsAAAAJ&hl=en)<sup>2</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>3</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Shanghai Jiao Tong University, <sup>3 </sup>Mayo Clinic <br/>
Published in: **International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2024](https://conferences.miccai.org/2024/en/))**

[Paper](https://papers.miccai.org/miccai-2024/paper/0271_paper.pdf) |  [Code](https://github.com/jlianglab/ASA) | [Poster] | [Slides] | Presentation ([YouTube])





## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@inproceedings{pang2024asa,
  title={Asa: Learning anatomical consistency, sub-volume spatial relationships and fine-grained appearance for ct images},
  author={Pang, Jiaxuan and Ma, DongAo and Zhou, Ziyu and Gotway, Michael B and Liang, Jianming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={91--101},
  year={2024},
  organization={Springer}
}

```

## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided in part by the ASU Research Computing and in part by the Bridges-2 at Pittsburgh Supercomputing Center through allocation BCS190015 and the Anvil at Purdue University through allocation MED220025 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also acknowledge Google for granting us access to CXR Foundation API, which enabled us to generate the embeddings for the target datasets. The content of this paper is covered by patents pending.


## License

Released under the [ASU GitHub Project License](./LICENSE).
