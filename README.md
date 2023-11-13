# DRL-Building-Energy-Ctr

This repository "Deep Reinforcement Learning Building Energy Control" hosts the source code for a recurrent reinforcement learning agent, specifically tailored for Home Energy Management Systems (HEMS). The agent is trained using a Gym environment based on the CoSES ProHMo Modelica framework. The primary focus of this agent is to efficiently control a building's heat pump and a three-way valve of a thermal storage.
The objective is twofold: to adhere to predefined thermal constraints and to optimize the process with a focus on minimizing electricity costs.


## Getting started

**Install libraries:**
```
pip install torch numpy pandas gymnasium pyfmi tensorflow
```

**Start policy training:**
```
python ./src/train_rsac.py
```

## Acknowledgments

- The building simulation was taken from [CoSES_thermal_ProHMo Public](https://github.com/DZinsmeister/CoSES_thermal_ProHMo). Thanks to DZinsmeister for his work.
- Portions of the code, especially those related to RSAC, are adapted from [off-policy-continuous-control](https://github.com/zhihanyang2022/off-policy-continuous-control/tree/pub). Thanks to zhihanyang2022 for his work.


## Citation

[Link to the IEEE Xplore publication](https://ieeexplore.ieee.org/document/10202844)

If you use this code or find it helpful for your research, please consider citing our publication:

```bibtex
@INPROCEEDINGS{10202844,
  author={Ludolfinger, Ulrich and Zinsmeister, Daniel and Perić, Vedran S. and Hamacher, Thomas and Hauke, Sascha and Martens, Maren},
  booktitle={2023 IEEE Belgrade PowerTech}, 
  title={Recurrent Soft Actor Critic Reinforcement Learning for Demand Response Problems}, 
  year={2023},
  pages={1-6},
  doi={10.1109/PowerTech55446.2023.10202844}
}
```

