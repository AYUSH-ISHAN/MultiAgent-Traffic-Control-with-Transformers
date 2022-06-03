# MultiAgent Traffic Control Using Transformers

Recent advances in multi-agent reinforcement learning have been largely limited training one model from scratch for every new task. This limitation occurs due to the restriction of the model architecture related to fixed input and output di- mensions, which hinder the experience accumulation and transfer of the learned agent over tasks across diverse levels of difficulty. Among these Multi-Agent Traffic Control is a hot topic research. So, here I have implemented the **UNIVERSAL MULTI - AGENT REINFORCEMENT LEARNING VIA POLICY DECOUPLING WITH TRANSFORMERS (UPDET)** on a custom build Multiagent Traffic Env.

# Environment Used

As said before the enviroment which I have used is a custom build environemnt of Traffic control with varied difficulities and number of agents. The demo of the env is shown below.

<p align="center"><img src="./assets/traffic.gif" height = "400" width="800"/></p>

In this project I have kept number of agent = **10** and diffulty as **medium**. One can change the difficulty and number of agents <a href="">here (add link to file here)<a/>.

# Model Architecture

<p align="center"><img src="./assets/algo.jpg"/><br><em>Diagram representing the network architecture used in the training</em></p>
<br>
The right part of he figure is the UPDeT algorithm which is a replacement of the traditional RNN/LSTM based base model. The main reason of this replacement is the advantages of transformers based model especially UPDET as compared to the RNN/LSTM based models as explained in the paper[1].<br>
The main advantages of this are : <br>
<ol>
<li>UPDeT-based MARL framework outperforms RNN-based frameworks by a large margin in terms of final performance on state-of-the-art centralized functions.<br></li>
<li>This model has strong transfer capability and can handle a number of different tasks at a time.</li>
<li>The model accelerates the transfer learning speed (total steps cost) to make it roughly 10 times faster compared to RNN-based models in most scenarios.</li>
</ol>
The left side of the figure is the Qmix mixer architecture as shown in paper[2].

# Result and Inference :











# References :

**[1]** <a href="https://arxiv.org/abs/2101.08001">UPDeT</a>: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers Siyi Hu, Fengda Zhu, Xiaojun Chang, Xiaodan Liang<br>
**[2]** <a href="https://arxiv.org/abs/1803.11485">QMix</a>: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson<br>
**[3]** <a href="https://arxiv.org/abs/1705.08926">COMA<a/>: Counterfactual Multi-Agent Policy Gradients Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson<br>
**[4]** <a href="https://arxiv.org/abs/1905.05408">QTran</a>: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, Yung Yi<br>
**[5]** <a href="https://arxiv.org/pdf/1903.04527.pdf">IQL</a>: Multi-Agent Deep Reinforcement Learning for Large-scale Traffic Signal Control Tianshu Chu, Jie Wang, Lara Codecà, and Zhaojian Li <br>
**[6]** <a href="https://arxiv.org/abs/1706.05296">VDN<a/>: Value-Decomposition Networks For Cooperative Multi-Agent Learning Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, Thore Graepel <br>
**[7]** Attention Is All You Need: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin <br>
**[8]** Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks Linghui Meng, Muning Wen, Yaodong Yang, Chenyang Le, Xiyun Li, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, Bo Xu
