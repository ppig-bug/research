# 论文注解

## 第一次：略读
1. 标题： 通过动作扩散进行视觉运动策略学习
2. 摘要部分：简单介绍了diffusion-policy是一个条件去噪扩散过程，即生成机器人的行为动作，用神经网络估计动作分布的对数概率梯度的和通过一系列动力学步骤迭代生成动作
3. 介绍部分：优点：解决多模态分布问题；一连串的未来动作的序列更有前瞻性，而不是局限于几步；训练的稳定性
三个核心设计：
- closed-loop action sequences:把预测高维度动作序列和滚动时域控制结合起来，既能闭环的不断重新规划，还能保证时间动作一致性
- visual conditioning: 把视觉当作扩散过程的条件，这样可以避免每次去噪迭代中反复计算，推进推理实时动作
- Time-series diffusion transformer 基于transformer的时间序列扩散网络，更加适应高频的动作变化，避免CNN模型太平滑的问题
4. 基本了解了每节的标题
5. 结论：优点稳定易于训练，但是也会受一些客观因素影响

## 第二步细读
1. 只关注图表部分： 
### fig1: 
对比explicit policy ,implicit policy ,diffusion policy :(b)图里面的energy可以理解为一个能量场去表示动作分布，能量越低，动作概率越高；(c)图可以看出 diffusion policy 是以o观测作为条件，然后去学习一个score 噪声场，从噪声出发迭代采样出动作
```
对比三种策略的表示方式：
explicit policy : 
输入观测o 直接输出动作  --结果会平均多模态，达不到理想的效果
implicit policy :
输入观测o，动作a,输出能量，动作需要靠优化 ---在a的基础上通过能量场来实现多模态的表示，但是energy是一个非凸函数，动作优化可能会卡在某个极小值上面
diffusion policy ：
输出噪声，动作靠的是多步去噪采样得到 ---在a,b基础上面改变方法：从高噪声的平滑地形开始，沿着不同噪声尺度下的score逐步的细化；训练比a,b更简单
```

### fig2：
diffusion policy overview
-  整个diffusion框架：输入观察序列，输出动作序列
-  依据最近一步观测O输出动作序列，再加入随机噪声k得到A* ，prediction Horizon Tp就是策略一次预测未来多少个动作，也就是输出动作序列的长度
- 同一个diffusion policy 两种去噪网络（充当去噪器）的实现：
- CNN-base整体流程如下（简单来说就是把动作序列看成时间序列，用1D卷积提取局部时间模式，输出每个时间步的噪声）
```
O_t -> ResNet-> obs_feat
k扩散步-> 正弦编码+MLP ->step_feat
A^k ->1D时间卷积 ->预测噪声εθ
与此同时 FiLM把观测和扩散步注入CNN的每一层

```
- Transformer-based整体流程如下:简单来说就是把动作序列变成token，用self-attention建模动作之间的关系，用cross-attention让每个动作查询观测到的信息
```
O_t -> ResNet-> obs_token
A^k ->linear+位置编码->action_tokens
k->step_embedding
        self_attention:动作token之间，a_t可以关注a{t+1},a{t+2},可以保证捕捉到快速的变化和和高频的动作
        cross-attention:动作的token查询观测的token，可以让每一个动作步查询到此刻的视觉信息，比FiLM更加灵活

最后得到预测噪声 εθ[T_p, D_a]

```
- 梳理一下思路：无论是CNN还是Transformer都充当一个去噪器，diffusion里面的网络：（A*k,k,o）->网络->εθ[T_p, D_a](这段动作里面混入的噪声估计)
```
观测o:告诉网络当时的场景
A^k:让网络帮忙找一下手里面这段噪声在哪个地方
k:告诉网络这段噪声有多重
网络：回答在这些位置混入了这段噪声
外部公式拿着这个噪声估计去更新A*K--- 多次网络调用+更新的公式就得出来了生成的动作

```

### fig3 
multimodal behavior:
- 图片将四种方法的选择路径形象化，可以明显发现只有diffusion policy可以从多种都可行的选择中选择出最优的一种，LSTM-GMM以及IBC只能偏执的选择其中一种

### fig4
Velocity v.s. position control 策略输出的动作是目标位置（例如：关节目标角）还是目标速度（关节速度，以某个速度运动）
- 横坐标是两种任务实验，一个是square（robotmmic抓取一个方形螺母把他套在杆子上面，讲求位置精度），另一个是kitchen P4（kitchen包含的几个长时程操作任务）
- 纵坐标是任务完成度
- fig4目的就是看Velocity v.s. position control在不同任务上面的表现，可以发现diffusion-policy比其他优势更突出

### fig5
diffusion policy消融实验
- x轴 left:Action Horizon t_a 每次预测出一段动作后，实际执行多少步再重新规划，t_a小就是频繁重新规划，动作容易抖，t_a大就是一次执行更长的动作，但是反应会慢；right：Latency鲁棒性：就是从最后一帧观测到第一个动作实际的执行中间隔了多少个控制步
- y轴：Y=当前设置的成功概率-该任务能达到的最高成功率；当y=0就达到了该任务的最佳表现，<0的分数就表示比最好的查理多少
- 两条折线：push-T（就是一个二维推T型方块到目标位姿的视觉操作）还有square
- 结论：ACtion horizon不是越大越好或者越小越好，在折中点的时候效果最好；精细装配任务对时序误差更加敏感
- **启发**：自己做实验的时候 action horizon要选中间值，不要太极端；精细的任务类似square这一类更怕实验误差，在做机器人灵巧手的实验时，属于精细的实验，一定要考虑时序问题，不同动作阶段也要考虑合适的horizon
