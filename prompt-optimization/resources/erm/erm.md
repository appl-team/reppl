Efficient and Accurate Prompt Optimization:
the Benefit of Memory in Exemplar-Guided Reflection

Cilin Yan1∗, Jingyun Wang1∗, Lin Zhang2∗, Ruihui Zhao2, Xiaopu Wu2,
Kai Xiong2, Qingsong Liu2, Guoliang Kang1†, Yangyang Kang23†,

1Beihang University, 2ByteDance China 3Zhejiang University

Abstract

Automatic prompt engineering aims to enhance
the generation quality of large language models
(LLMs). Recent works utilize feedbacks gener-
ated from erroneous cases to guide the prompt
optimization. During inference, they may fur-
ther retrieve several semantically-related exem-
plars and concatenate them to the optimized
prompts to improve the performance. How-
ever, those works only utilize the feedback at
the current step, ignoring historical and uns-
eleccted feedbacks which are potentially ben-
eficial. Moreover, the selection of exemplars
only considers the general semantic relationship
and may not be optimal in terms of task perfor-
mance and matching with the optimized prompt.
In this work, we propose an Exemplar-Guided
Reflection with Memory mechanism (ERM) to
realize more efficient and accurate prompt opti-
mization. Specifically, we design an exemplar-
guided reflection mechanism where the feedback
generation is additionally guided by the gener-
ated exemplars. We further build two kinds of
memory to fully utilize the historical feedback
information and support more effective exem-
plar retrieval. Empirical evaluations show our
method surpasses previous state-of-the-arts with
less optimization steps, i.e., improving F1 score
by 10.1 on LIAR dataset, and reducing half of
the optimization steps on ProTeGi.

1

Introduction

Prompt optimization is crucial for enhanc-
ing the performance of Large Language Mod-
els (LLMs). Even a subtle adjustment to the

∗Equal contribution. †Corresponding author.

Jingyun Wang and Guoliang Kang
Cilin Yan,
are with Beihang University(e-mail:
{clyan,
19231136}@buaa.edu.cn, kgl.prml@gmail.com), and Lin
Zhang, Jingyun Wang, Ruihui Zhao, Xiaopu Wu, Kai
Xiong, Qingsong Liu are with ByteDance China(e-mail:
xiongkai.kx,
zhaoruihui, wuxiaopu,
{zhanglin.hb,
liuqingsong}@bytedance.com , and Yangyang Kang is
with ByteDance China and Zhejiang University(e-mail:
yangyangkang@bytedance.com).

prompt may lead to an obvious improvement
or decline in performance, thereby highlight-
ing the critical role of prompt engineering for
LLMs. Manual prompt engineering demands
significant human effort and expert knowledge,
while traditional fine-tuning methods (Lester
et al., 2021; Shin et al., 2020) heavily rely on
substantial computational resources and power-
ful GPUs. Therefore, it is necessary to explore
automatic prompt engineering, which is com-
patible with black-box APIs (e.g., GPT-4) and
does not require extensive resources.

Recently, feedback-based methods (Ye et al.,
2023; Juneja et al., 2024) exhibit promising
performance for automatic prompt engineering,
which generally leverage feedbacks generated
from failure cases to facilitate the prompt op-
timization process. Previous feedback-based
methods have two main drawbacks. Firstly,
they throw unselected and historical feedbacks
which may benefit the prompt optimization, re-
sulting in more optimization steps to achieve
satisfactory performance. Secondly, during in-
ference, previous methods may retrieve several
semantically-related exempalrs and concatenate
them to the optimized prompt to improve the
performance. However, the retrieved exemplars
are not optimal without evaluating their influ-
ence on the task performance. Those draw-
backs largely constrain both the accuracy and
efficiency of the prompt optimization process.
In this work, we introduce an Exemplar-
Guided Reflection with Memory mecha-
nism (ERM) to achieve efficient and accurate
prompt optimization. Firstly, we propose an
exemplar-guided reflection mechanism. We
manually design an instructive meta-prompt.
Unlike previous meta-prompts which simply
guide LLMs to reflect on the current case, our in-
structive meta-prompt further directs LLMs to
generate exemplars by selecting typical wrong

Figure 1: Feedback-based automatic prompt engineering methods commonly employ a meta-prompt , which
guides LLMs to evaluate the current case, provide feedbacks , and generate refined prompts . In this work,
we design an instructive meta-prompt to select exemplars with detailed solution processes, and generate
feedbacks for the current case. These feedbacks are stored in Feedback Memory and periodically retrieved
to efficiently guide the optimization of prompts . Additionally, these exemplars are stored and assessed in
an Exemplar Factory to enhance prediction accuracy.

samples and providing detailed solution pro-
cesses for them. Due to the detailed solution
processes within exemplars, LLMs therefore
yield more informative feedbacks. We then pro-
pose a Feedback Memory to store all feedbacks
and assign a priority score to each of them.
During the optimization process, we retrieve
a group of feedbacks with the highest priority
scores and instruct LLMs to generate a new
prompt for the feedbacks. After evaluating the
refined prompts, we update the priority scores
of the associated feedbacks accordingly, i.e., we
increase the score for improved performance
and decrease it if no gain. Consequently, feed-
backs with valuable insights will be consistently
selected rather than ignored throughout the op-
timization process. We store all exemplars in
an Exemplar Factory and assign a prior score
to each piece of them. At the inference stage,
we retrieve a set of exemplars with the highest
priority score, and concatenate the exemplars
to our refined prompt to further improve the
performance of LLM.

We conduct an extensive evaluation on seven
tasks to compare our method with the lat-
est prompt optimization approaches. Our re-
sults demonstrate substantial improvements
over state-of-the-art methods, notably achiev-
ing a 10.1 improvement in F1 score on the LIAR
dataset. Additionally, the optimization speed of
our method is roughly twice as fast as ProTeGi.
Our contributions are summarized as follows:
1) We design an instructive meta-prompt,

which guides LLMs to select exemplars and
therefore yield more informative feedbacks.

2) We propose a Feedback Memory to store
historical feedbacks by their priority scores, en-
abling effective retrieval and utilization of feed-
backs for prompt optimization.

3) We propose an Exemplar Factory to store
and evaluate exemplars. By retrieving exem-
plars and concatenating them to our refined
prompt at the inference stage, we further en-
hance the performance of LLM.

4) We conduct extensive experiments on var-
ious tasks and show superior performance of
our method to previous state-of-the-arts. Addi-
tionally, our optimization steps can be largely
reduced, e.g., the steps of our method are ap-
proximately half of that in ProTeGi.

2 Related Work

2.1 Automatic Prompt Optimization

Prompt engineering (Zhou et al., 2022) aims
to identify suitable prompts as inputs for
large language models (LLMs) to perform var-
ious tasks. To minimize human effort, re-
searchers have explored automatic prompt opti-
mization (Lester et al., 2021; Shin et al., 2020;
Li and Liang, 2021). Previous works adopt
various strategies for automatic prompt opti-
mization, such as evolutionary-based methods,
trajectory-based methods, and feedback-based
methods. Evolutionary-based methods (Guo
et al., 2024; Fernando et al., 2024) utilize LLMs
to rewrite a set of prompts using evolutionary al-

(a) Exemplar-Guided ReflectionFind 4 varied examples with step-by-step solutions where this prompt fails. Then, provide 3 reasons for the failure.Q: Elizabeth uses $3.00 worth of ingredients to make a  bag of granola.  She makes 20 bags and sells them for $6.00 .... A: 1. Calculate the total cost of ingredients: 20 bags * $3.00 = $60.00. 2. Calculate the revenue from selling the first 15 bags...Feedback: The prompt should emphasize breaking down the problem into smaller, manageable steps to ensure all intermediate calculations are included.Score: 0.7Score: 0.8Feedback Memory Storage(b) Feedback MemoryRetrievalUpdateFeedback: The prompt should emphasize breaking down the problem into smaller, manageable steps to ensure all intermediate calculations are included.Feedback: The prompt should encourage re-checking the final calculations to avoid minor arithmetic mistakes that can significantly affect the final answer.Feedback: The prompt should emphasize breaking down the problem into smaller...Refine the prompt so that the model predicts correctly based on the above information.Ensure you understand the problem and carefully perform each calculation. After obtaining the answer, double-check your work to verify the accuracy of the final result.Score: 0.8Exemplar Memory Storage Score: 0.9Q: Liam wants to go to Paris, but first, he has to pay his bills. His trip costs $7,000... The answer is 1,500.Q: Steve has a bank account that ...A: . Calculate the amount after the first year: $100 * 1.10 + $10 = $120. 2. Calculate...RetrievalUpdateQ: Elizabeth uses $3.00 worth of ingredients...A: 1. Calculate the total cost of ingredients: 20 bags * $3.00 = $60.00. 2. Calculate the... Carefully break down the problem into smaller, manageable parts and ensure...Q: Elizabeth uses $3.00 worth of ingredients... A: 1. Calculate the total cost of ingredients...(c) Exemplar FactoryFigure 2: Pipeline of ERM. In wrong prediction samples, the instructive reflective meta-prompt is employed
to select exemplars with detailed answer processes, which are subsequently followed by feedback generation.
The feedbacks are stored in feedback memory storage, and the exemplars are stored in exemplar memory
storage. These stored feedbacks are periodically retrieved to efficiently guide prompt optimization, with
selective forgetting based on their effectiveness in enhancing optimization. Additionally, these exemplars
are assessed to enhance prediction accuracy.

gorithms (Holland, 1992; Storn and Price, 1997),
selecting the best ones on a validation set to sim-
ulate the natural selection process for optimiz-
ing prompts. Trajectory-based methods (Yang
et al., 2024; Tang et al., 2024) employ an LLM
prompt optimizer to generate new prompts
based on historical prompts, scores, or error
examples. Feedback-based methods (Pryzant
et al.; Juneja et al., 2024) use LLMs to summa-
rize feedback on erroneous cases, leveraging this
feedback to optimize and create new prompts.
In this work, we focus primarily on feedback-
based methods, aiming to write stronger feed-
back and efficiently utilize it for optimization.

2.2 Long-Term Memory Mechanisms

Existing automatic prompt optimization meth-
ods (Pryzant et al.; Juneja et al., 2024) face
challenges in maintaining a robust long-term
memory function, limiting their ability to retain
and utilize valuable feedback for prompt opti-
mization. MemoryBank (Zhong et al., 2024)
solves the challenge of maintaining a robust
long-term memory conversation history in pre-
vious LLMs (Touvron et al., 2023; Zeng et al.,
2022; Taori et al., 2023) by introducing a mech-
anism that enhances their ability to store and
recall relevant information over time. This
approach mimics human memory dynamics
through a selective retention strategy inspired
by the Ebbinghaus Forgetting Curve (Ebbing-
haus, 2013). Our work builds on these advance-

ments by using memory storage to implement
feedbacks and exemplars in long-term mem-
ory. We implement a forgetting strategy for
feedbacks and exemplars that are retrieved but
deemed unvaluable, thereby enhancing the ef-
ficiency and accuracy of long-term memory re-
tention in prompt optimization.

3 Method

In this section, we propose ERM, a novel
method designed to achieve efficient and accu-
rate prompt optimization. As shown in Figure 2,
ERM is composed of three core components: (1)
Exemplar-Guided Reflection, employing an
instructive meta-prompt (Section 3.2), guides
prompt optimizer to first generate exemplars
by identifying typical wrong samples and pro-
viding detailed solution processes, followed by
generating feedback. (2) We then propose a
Feedback Memory (Section 3.3) to store all
feedbacks and assign a priority score to each
piece of them. These feedbacks can then be
retrieved and utilized during optimization effi-
ciently. After evaluating the refined prompts,
we update the priority scores of the associated
feedbacks. (3) Finally, we utilize an Exemplar
Factory (Section 3.4) to store and evaluate
exemplars, which serve as additional resources
during prediction. By incorporating the re-
trieved exemplars into our refined prompt, task
model are further guided to achieve improved

# TaskSolve the math problem.# Exemplars{anchor_exam# PredictionInput: {question} Output: I'm trying to write a math problem solver prompt.The current prompt is: {prompt}Here are some valuable pieces of feedback:{feedback_1}{feedback_2}Based on this information, please write a better prompt.Sure, these are the typical examples:{exemplar_1}{exemplar_2}Considering the error cases and typical examples, the prompt should be improved as follows:{feedback_1}{feedback_2}I'm trying to write a math problem solver prompt.The current prompt is: {prompt}But this prompt gets the following examples wrong:{error_samples}Please identify some typical examples with detailed solutions from the cases above where the current prompt fails, to help improve my understanding and performance. These examples should be diverse to cover a range of different issues.After identifying these typical examples, please provide some reasons why the prompt could have gotten these examples wrong.Feedback Memory StorageExemplar Memory Storage{feedback_1}{feedback_2}{feedback_i}{score_1}{score_2}{score_i}{exemplar_1}{exemplar_2}{exemplar_i}{score_1}{score_2}{score_i}(a) Exemplar-Guided ReflectionGeorge had 28 socks... So, the answer is 64.Feedback Forgetting UpdateAre the feedbacks helpful?(b) Feedback Memory(c) Exemplar FactoryExemplar Forgetting UpdateAre the exemplars helpful?Approach each mathematical problem calmly and methodically. Identify the key information ...{exemplar_1}{exemplar_2}{prompt}{prompt}{feedback_1}{feedback_2}promptmeta-promptfeedbackexemplaraccuracy.

3.1 Preliminary
Given a training set Dtrain = {(qi, ai)}n
i=1 (qi
represents the question and ai is the paired an-
swer) and a test set Dtest drawn from a specific
task, along with a score function s(·) for this
task, we aim to perform the task using a black-
box task model Ms (e.g., ChatGPT), which
combines the prompt p with questions from
Dtest as input to generate responses. These
responses are then evaluated by the score func-
tion to calculate an average score over Dtest.
The goal of prompt optimization is to find an
optimal prompt p∗ drawn from the natural lan-
guage space that maximizes the expectation of
the average score over Dtest:

p∗ = arg max

p

E(qi,ai)∼Dtest[s(Ms(qi; p), ai)],

(1)
where p = [pI , pR(qi)] might be composed of
two parts: one includes the invariant content
pI , which remains independent of the question
and may include task descriptions and general
solution steps, and the other is the variable
content pR(qi) , which is question-specific. We
leverage a more powerful prompt optimizer Me
(e.g., GPT-4)compared with the task model
Ms to summarize feedbacks and optimize the
prompt.

Previous work typically divides prompt opti-
mization into three steps: prompt initialization,
new prompt proposal, and prompt search.

(1) Prompt initialization. Prompt initial-
ization can be achieved by both manual initial-
ization and induction initialization. Following
ProTeGi (Pryzant et al.), we initialize the orig-
inal prompt p0 manually.

(2) New prompt proposal. Commonly,
previous methods use a prompt optimizer to
summarize errors from the wrong samples
B = (˜qi, ˜ai), where the response of task model
Ms(˜qi, pt) is different from ˜ai, and then generate
feedbacks F t = Me(pt, B; pmeta
ref ) accordingly,
where pmeta
is the meta-prompt guiding the
prompt optimizer to generate feedback. The
prompt optimizer then optimizes and refines
the prompt pt based on the feedbacks to obtain
refined prompts pt+1 = Me(pt, B, f t; pmeta
opt ),
where f t ∈ F t, and pmeta
is the meta-prompt
opt
guiding the prompt optimizer to propose refined
prompt.

ref

(3) Prompt search. Following ProTeGi, we
employ a beam search strategy to further select
the refined prompts. Among several candidate
prompts P t+1, we select k prompts which per-
form best on the validation set, which is the
subset of the training set. These k prompts are
then used for the next optimization step.

3.2 Exemplar-Guided Reflection

To encourage the prompt optimizer generate
more informative feedbacks, we propose an
Exemplar-Guided Reflection in Figure 2(a),
which utilizes an instructive meta-prompt to
select typical wrong samples with detailed solu-
tion processes as exemplars and generate feed-
backs for them. Detailedly, we first utilize the
instructive meta-prompt pmeta
ref ∗ , which guides
the prompt optimizer to select m diverse and
significantly representative wrong samples from
the wrong samples B as exemplars E t and pro-
vide detailed solution processes for them:

E t = Me(pt, B; pmeta

ref ∗ ),

(2)

i=1 = {(˜qi, ˜ai, ˜coti)}m

where E t = {ei}m
i=1 is a set
of exemplars ei with detailed solution processes
˜coti. Then, the prompt optimizer generates
feedbacks F t = {f t}nf
, which offer insights on
i
example predictions and suggestions on modifi-
cation of the prompt:

F t = Me(pt, B, E t; pmeta

ref ∗ ),

(3)

Based on the wrong samples B and each item
in the generated feedbacks f t ∈ F t, the model
finally produce a refined prompt pt+1 for each
feedback:

pt+1 = Me(pt, B, f t; pmeta

opt ).

(4)

3.3 Feedback Memory

Aiming to accelerate the convergence of prompt
optimization process, we propose a Feedback
Memory in Figure 2(b). We store the feedbacks
with priority scores via a long-term memory
mechanism and retrieve them efficiently for opti-
mization. By evaluating the generated prompts,
we selectively forget the feedbacks to ensure
that all stored feedbacks remain beneficial for
prompt optimization.
Feedback Memory Storage In Feedback
Memory, we store the valuable feedbacks during
the optimization process and assign a priority

score to each piece of them, which serves as a
basic foundation for Feedback Forgetting Up-
dating. To effectively store useful feedbacks and
prevent adverse impacts on prompt optimiza-
tion, we employ a feedback filtering strategy:
(1) We evaluate the refined prompts generated
based on the feedbacks, and only store the infor-
mative feedbacks whose corresponding prompts
bring improvements on the validation set. Such
strategy ensures that only valuable feedbacks
are stored and retrieved. (2) Additionally, we
employ the BGE-M3 model (Chen et al., 2024)
to calculate the semantic similarity between
newly generated feedbacks and the stored ones.
We ignore the feedbacks of high similarity with
the previous ones to avoid redundant informa-
tion.
Feedback Retrieval During the optimization
process, we periodically select historical feed-
backs from the memory based on their priority
scores. Specifically, we calculate the selection
probability for each feedback as follows:



Pf = softmax



)| ˜F|

(
e

sp(fi)
τf





i=1

(5)

where τf is the temperature, controlling the
tendency to select high-scoring feedbacks, and
˜F denotes all feedbacks stored in the memory.
We then randomly select n feedbacks according
to their selection probabilities:

{f }n

i = sample( ˜F, Pf )

(6)

Feedback Forgetting Updating The selected
feedbacks {f }n
i guide prompt optimizer gener-
ate new prompts pt+1 = Me(pt, B, {f }n
i ; pmeta
opt∗ ),
where pmeta
opt∗ is the meta-prompt that efficiently
utilizes the feedback group to generate a re-
fined prompt. We then update their priority
scores by evaluating the generated prompt: we
increase the priority score if the performance is
improved but decrease it if no gain.

(7)

p(f ) = (1 − β)sp(f )t−1 + βI(f )
st
where I(f ) represents whether sufficient per-
formance gain is achieved and β is a hyper-
parameter to control the speed of updating.
Besides, the feedback will be removed from the
storage once its priority score falls below a cer-
tain threshold t:

˜F t = f ∈ ˜F t−1 | st

p(f ) ≥ t.

(8)

With such Forgetting Updating mechanism, we
ensure that the most valuable feedbacks are con-
tinuously utilized, which efficiently accelerate
the convergence of our optimization process.

3.4 Exemplar Factory

As shown in Figure 2(c), we store the exemplars
along with a priority score to each piece of them,
similar to that in Feedback Memory. These
exemplars are stored in memory and retrieved
for prediction, allowing us to assess their impact
on the task. We selectively forget exemplars,
ensuring that the valuable ones will be retrieved
to enhance the prediction performance.
Exemplar Memory Storage The exemplar
memory storage retains valuable exemplars. We
introduce an exemplar filtering strategy to en-
sure stored exemplars benefit prediction: (1)
We verify that the detailed solution process of
the exemplar generated by prompt optimizer
matches to the ground truth label. (2) When
a new generated exemplar is identical to the
stored ones, we replace the stored exemplars
with probability p and reject the new exem-
plar with probability 1 − p to avoid redundant
storage.
is
Exemplar Retrieval Each exemplar ei
assigned a priority score sp(ei). During the
prompt optimization process, we calculate the
selection probability for each exemplar as fol-
lows:

Pe = softmax





(
e

sp(ei)·s
τe

)| ˆE|

j
s(ei)


 (9)

i=1

where sp(ei) is the priority score of exemplar
ei and sj
s(ei) is its semantic similarity to the
question j, ˆE represents the stored exemplars,
and τe is the temperature. We then randomly
sample five exemplars as variable content pR(qi)
of prompt. During the inference stage, we select
the five exemplars with the highest se(ei) · sj
s(ei)
as variable content pR(qi) of prompt for more
accurate predictions.
Exemplar Forgetting Updating We ad-
just the priority scores of exemplars based on
whether incorporating them as the variable con-
tent pR(qi) in the prompt leads to improve-
ments. Exemplars with low priority scores are
promptly removed to ensure that only valuable
ones are stored.

True / False

Generative

Multiple-choice

Method

LIAR
(F1)

BBH
(F1)

ETHOS
(F1)

ArSarcasm
(F1)

WebNLG
(Rouge-L)

GSM8K
(Acc.)

Empty
CoT (Kojima et al., 2022)

46.4
46.0

47.7
APE (Zhou et al., 2022)
58.5
ProTeGi (Pryzant et al.)
OPRO (Yang et al., 2024)
47.9
Promptbreeder (Fernando et al., 2024) 47.1
47.9
EvoPrompt (Guo et al., 2024)
54.7
GPO (Tang et al., 2024)

69.4
81.9

72.9
73.6
75.7
74.3
75.0
70.8

93.0
84.5

94.0
96.5
93.5
94.5
93.0
94.0

ERM
∆

68.6 86.1
98.0
+10.1 +4.2 +1.5

83.7
83.7

83.8
84.1
84.5
83.8
83.8
83.6

85.1
+0.6

49.4
49.3

51.3
55.7
51.9
51.0
50.2
51.8

59.6
+3.9

89.0
89.0

91.3
91.0
90.7
91.7
90.7
90.3

93.3
+1.6

WSC
(Acc.)

77.3
81.3

79.3
80.0
83.3
80.0
78.8
84.0

86.0
+2.0

Table 1: Comparisons of our method with existing LLM-based prompt optimizers under zero-shot setting.

Method

Prompt

Empty
ProTeGi

Write the following triples as fluent English text.
You are given a set of triples that need to be converted into coherent and fluent English
sentences. Each triple consists of a subject, predicate, and object. Your task is to
accurately convey the information from these triples into well-formed sentences. Ensure
the sentences are complete, grammatically correct, and clearly express the relationships
provided in the triples.
OPRO
Convert the following sets of triples into coherent, natural, and fluent English sentences.
PromptBreeder Transform these triples into smooth and stylish English sentences, and make them shine!
Turn the provided triples into smooth, flowing English sentences that will impress
EvoPrompt
everyone!
Rewrite these triples into fluent and natural English sentences.
Convert the following triples into coherent and fluent English sentences. Ensure that all
relationships and attributes are accurately conveyed. When multiple associations or
attributes are involved, break down the information into smaller, logical sentences to
maintain clarity.

GPO
ERM

Rouge-L

49.4
55.7

51.9
50.9
50.2

51.8
59.6

Table 2: Prompts optimized by different methods on the WebNLG dataset.

4 Experiments

Datasets. We perform evaluation on 7 stan-
dard datasets : WSC (Levesque et al., 2012),
Ethos (Mollas et al., 2022), ArSarcasm (Farha
and Magdy, 2020), Liar (Wang, 2017), BBH-
navigate (Suzgun et al., 2022), GSM8k (Cobbe
et al., 2021), WebNLG (Gardent et al., 2017).
Among these, ArSarcasm, Ethos, and Liar,
and BBH-navigate contain true/false ques-
tions, WSC contains multiple-choice questions,
GSM8K contains questions with integer an-
swers, and WebNLG contains questions requir-
ing natural language generation.
Baselines. We compare several representa-
tive methods, including existing LLM-based
prompt optimizers: APE (Zhou et al., 2022),
APO (Pryzant et al.), OPRO (Yang et al., 2024),
Promptbreeder (Fernando et al., 2024), Evo-
Prompt (Guo et al., 2024), and GPO (Tang
et al., 2024). In addition, we consider the base-

line using manually written simple prompts
(“Manual”), which we provide in the appendix,
and the instruction “Let’s think step by step.”
from chain-of-thought prompting (“CoT”) (Ko-
jima et al., 2022) for performance comparison.
Evaluation Metrics. We report the F1 score
on Ethos, ArSarcasm, Liar and BBH-navigate
following (Pryzant et al.), accuracy on WSC
and GSM8k following (Tang et al., 2024; Juneja
et al., 2024) and ROUGE-L on WebNLG fol-
lowing (Tang et al., 2024).
Implementation Details. For the task model,
we use Doubao-Pro (ByteDance, 2024). For the
prompt optimizer, we use GPT-4o (OpenAI,
2024). We repeat all the experiments three
times and report the average of the results.
Other details are presented in appendix.

4.1 Main Results

Comparison under Zero-shot Setting. Ta-
ble 1 presents the results of different methods

Figure 3: The efficiency of our approach ERM. The size of the circles represents performance, with larger
circles indicating better performance. The vertical axis shows the optimization steps needed for different
methods to achieve peak performance across datasets.

True / False

Generative

Multiple-choice

Method

LIAR
(F1)

BBH
(F1)

ETHOS
(F1)

ArSarcasm
(F1)

WebNLG
(Rouge-L)

GSM8K
(Acc.)

51.2
APE (Zhou et al., 2022)
60.3
ProTeGi (Pryzant et al.)
52.1
OPRO (Yang et al., 2024)
Promptbreeder (Fernando et al., 2024) 51.8
52.3
EvoPrompt (Guo et al., 2024)
56.6
GPO (Tang et al., 2024)

74.3
73.6
75.0
75.7
76.4
75.0

93.2
97.0
94.8
95.7
94.3
95.5

84.3
84.1
84.7
84.5
83.9
83.8

53.1
56.3
52.4
52.7
51.8
53.4

91.8
91.0
90.8
91.7
90.9
90.5

ERM

68.6 86.1

98.0

85.1

59.6

93.3

WSC
(Acc.)

80.3
81.0
85.0
81.5
80.4
84.9

86.0

Table 3: Comparisons of our method with existing LLM-based prompt optimizers under few-shot setting.

Exemplar-Guided
Reflection

Feedback Memory Exemplar Factory

LIAR
(F1)

BBH
(F1)

ETHOS
(F1)

ArSarcasm
(F1)

WebNLG
(Rouge-L)

GSM8K
(Acc.)

WSC
(Acc.)

✓
✓
✓
✓

✓
✓

✓

✓

58.5
62.9
67.2
66.6
68.6

73.6
75.7
84.7
82.6
86.1

96.5
97.0
97.0
97.5
98.0

84.1
84.2
84.9
84.8
85.1

55.7
56.9
58.6
58.8
59.6

91.0
92.7
93.0
93.0
93.3

80.0
82.0
84.0
85.0
86.0

Table 4: Effect of each component in our method.

for prompt optimization across true/false ques-
tions, generative questions, and multiple-choice
questions.

For true/false questions, our method demon-
strates a significant improvement over previ-
ous works. Specifically, our method outper-
forms trajectory-based methods (OPRO and
GPO) by 13.9. Trajectory-based methods uti-
lize an LLM prompt optimizer to generate new
prompts based on historical prompts, scores,
or error examples, but may struggle to iden-
tify “better prompts”,
limiting their perfor-
mance. Our method also outperforms ProTeGi
(a feedback-based method) by 10.1, which can
be attributed to our proposed exemplar-guided
reflection, feedback memory and example fac-
tory.

For generative questions and multiple-choice
questions, our method also significantly outper-
forms previous methods. Specifically, on the
WebNLG dataset, our approach surpasses previ-
ous methods by 3.9 in Rouge-L score. Table 2 vi-
sualizes the optimized prompts on the WebNLG
dataset, demonstrating that our method’s opti-
mized prompts are more effective at capturing
the critical information needed to enhance task
performance. Specifically, the exemplar factory
contributes an F1 score improvement of 3.7 on
the LIAR dataset, while the feedback memory
results in a 2.0 improvement.

Efficiency of Our Method. Our approach
introduces a memory mechanism to efficiently
store and utilize feedbacks. We show the op-
timization steps needed for different methods

(a) LIAR789101112131415Optimized Steps(b) BBH78910111213(c) ETHOS891011121314(d) ArSarcasm7891011121314(e) WebNLG56789101112131415(f) GSM8K6789101112131415(g) WSC5678910111213APEProTeGiOPROPromptbreederEvoPromptGPOERMRetrieval

Exemplar
Filtering

Selective
Forget.

LIAR
(F1)

BBH
(F1)

WebNLG
(Rouge-L)

Retrieval

Feedback
Filtering

Selective
Forget.

LIAR
(F1)

BBH
(F1)

WebNLG
(Rouge-L)

✓
✓
✓

✓
✓

✓

62.9
62.3
65.7
66.6

75.7
75.0
81.3
82.6

56.9
57.0
58.4
58.8

✓
✓
✓

✓
✓

✓

66.6
66.4
67.5
68.6

82.6
81.9
82.6
86.1

58.8
58.8
59.2
59.6

Table 5: Effect of each component in Exemplar
Factory.

Table 6: Effect of each component in Feedback Mem-
ory.

to achieve peak performance across datasets
in Figure 3, which highlights the superior effi-
ciency of our method. Specifically, according to
Figure 3(a), on the LIAR dataset, our method
reaches an F1 score of 68.6 by the 7th step,
while ProTeGi only achieves 58.5 by the 13th
step, demonstrating that our method nearly
doubles the optimization speed.
Comparison under Few-shot Setting. Ta-
ble 3 presents a comparison between our method
and others under few-shot settings. For each
approach, we dynamically select five relevant
examples through k-nearest neighbors (kNN)
clustering in the embedding space. According
to the results, ERM consistently outperforms
the previous methods. Notably, on the LIAR
dataset, our approach achieves an 8.3 F1 score
improvement over previous methods, demon-
strating the effectiveness of selecting valuable
wrong examples as exemplars and equipping
them with chain-of-thought-like solution pro-
cesses.

4.2 Ablation Study

Effect of Each Component. In Table 4, we
conduct experiments to verify the effectiveness
of each key component in our method. We
adopt a strategy which dentify exemplars, con-
template the corresponding chain of thought
and then complete feedbacks, and observe that
ERM improves the F1 score by 4.4 on the LIAR
dataset compared with the approaches with-
out the instructive meta-prompt, which vali-
dates the effectiveness of the instructive meta-
prompt.] Additionally, the introduction of the
memory mechanism for feedback memory and
exemplar factory brought a further 5.7 improve-
ment on the LIAR dataset, confirming the ef-
fectiveness of the memory mechanism.
Effect of Exemplar Factory. As shown
in Table 5,
incorporating exemplar filtering
when storing exemplars does not enhance per-
formance. This is because the behavior of the

prompt optimizer is unpredictable and may gen-
erate incorrect or unconventional questions. Re-
trieving such examples does not enhance pre-
diction performance. However, filtering out er-
roneously generated exemplars and redundant
ones already in storage resulted in a 3.4 improve-
ment, highlighting the importance of exemplar
filtering. The introduction of a selective for-
getting further improved the F1 score by 0.9
on the LIAR dataset, as it removes exemplars
that do not aid in prediction, thereby enhancing
performance.

Effect of Feedback Memory. As shown
in Table 6, directly storing feedbacks for peri-
odic optimization without the feedback filtering
strategy does not improve performance. Intro-
ducing the filtering strategy increased the F1
score on the LIAR dataset by 0.9 compared
to not using stored feedbacks. Additionally,
incorporating selective forgetting, which dis-
cards suboptimal feedback promptly, further
enhanced the F1 score by an additional 0.9.

5 Conclusion

In this paper, we introduce Exemplar-Guided
Reflection with Memory mechanism (ERM),
a novel approach to achieve efficient and accu-
rate prompt optimization. Using a instructive
reflection meta-prompt, ERM instructs LLMs
to select exemplars with detailed solution pro-
cesses and generate stronger feedback. We then
propose Feedback Memory mechanism to ef-
ficiently exploit potentially valuable feedback.
Additionally, Exemplar Factory is introduced
to further enhance the accuracy of prediction
by pre-assessing the impact on the task. ERM
refines prompts authored by human experts and
outperforms established automatic prompt engi-
neering baselines across various scenarios, with
optimization steps approximately half of that
in previous work.

6 Limitations

In this work, we effectively utilize feedbacks and
exemplars using a long-term memory mecha-
nism. However, in real-world applications, we
encounter additional challenges: some questions
continue to be incorrectly answered during the
optimization process, and prompt optimization
doesn’t always align with human expectations.
When the model struggles to optimize, intro-
ducing human intervention might aid in en-
hancing prompt optimization. This paper lacks
exploration on how humans could assist in the
optimization process. For instance, with persis-
tent incorrect answers, human input could offer
crafted solutions, helping the expert model gen-
erate improved feedback. Additionally, due to
computational and budget constraints, our ex-
periments are limited to representative tasks.

References

Derek Austin and Elliott Chartock. 2024. Grad-
sum: Leveraging gradient summarization for
optimal prompt engineering.
arXiv preprint
arXiv:2407.12865.

ByteDance. 2024. Doubao.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. Bge
m3-embedding: Multi-lingual, multi-functionality,
multi-granularity text
through
arXiv preprint
self-knowledge distillation.
arXiv:2402.03216.

embeddings

Karl Cobbe, Vineet Kosaraju, Mohammad Bavar-
ian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton,
Reiichiro Nakano, et al. 2021. Training verifiers
to solve math word problems. arXiv preprint
arXiv:2110.14168.

Nicholas Crispino, Kyle Montgomery, Fankun Zeng,
Dawn Song, and Chenguang Wang. 2023. Agent
instructs large language models to be general zero-
shot reasoners. arXiv preprint arXiv:2310.03710.

Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yi-
han Wang, Han Guo, Tianmin Shu, Meng Song,
Eric P Xing, and Zhiting Hu. 2022. Rlprompt:
Optimizing discrete text prompts with reinforce-
ment learning. arXiv preprint arXiv:2205.12548.

Hermann Ebbinghaus. 2013. Memory: A contri-
bution to experimental psychology. Annals of
neurosciences, 20(4):155.

Ibrahim Abu Farha and Walid Magdy. 2020. From
arabic sentiment analysis to sarcasm detection:
The arsarcasm dataset. In Proceedings of the 4th

Workshop on Open-Source Arabic Corpora and
Processing Tools, with a Shared Task on Offensive
Language Detection, pages 32–39.

Chrisantha Fernando, Dylan Sunil Banarse, Henryk
Michalewski, Simon Osindero, and Tim Rock-
täschel. 2024. Promptbreeder: Self-referential
self-improvement via prompt evolution. In Forty-
first International Conference on Machine Learn-
ing.

Claire Gardent, Anastasia Shimorina, Shashi
Narayan, and Laura Perez-Beltrachini. 2017. Cre-
ating training corpora for nlg micro-planning. In
55th Annual Meeting of the Association for Com-
putational Linguistics, ACL 2017, pages 179–188.
Association for Computational Linguistics (ACL).

Qingyan Guo, Rui Wang, Junliang Guo, Bei Li,
Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian,
and Yujiu Yang. 2024. Connecting large language
models with evolutionary algorithms yields pow-
erful prompt optimizers. In The Twelfth Interna-
tional Conference on Learning Representations.

John H Holland. 1992. Genetic algorithms. Scien-

tific american, 267(1):66–73.

Gurusha Juneja, Nagarajan Natarajan, Hua Li, Jian
Jiao, and Amit Sharma. 2024. Task facet learning:
A structured approach to prompt optimization.
arXiv preprint arXiv:2406.10504.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid,
Yutaka Matsuo, and Yusuke Iwasawa. 2022.
Large language models are zero-shot reasoners.
Advances in neural information processing sys-
tems, 35:22199–22213.

Brian Lester, Rami Al-Rfou, and Noah Constant.
2021. The power of scale for parameter-efficient
prompt tuning. arXiv preprint arXiv:2104.08691.

Hector Levesque, Ernest Davis, and Leora Morgen-
stern. 2012. The winograd schema challenge. In
Thirteenth international conference on the princi-
ples of knowledge representation and reasoning.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:
Optimizing continuous prompts for generation.
arXiv preprint arXiv:2101.00190.

Yujian Betterest Li and Kai Wu. 2023. Spell: Se-
mantic prompt evolution based on a llm. arXiv
preprint arXiv:2310.01260.

Ruotian Ma, Xiaolei Wang, Xin Zhou, Jian Li, Nan
Du, Tao Gui, Qi Zhang, and Xuanjing Huang.
2024. Are large language models good prompt
optimizers? arXiv preprint arXiv:2402.02101.

Ioannis Mollas, Zoe Chrysopoulou, Stamatis Karlos,
and Grigorios Tsoumakas. 2022. Ethos: a multi-
label hate speech detection dataset. Complex &
Intelligent Systems, 8(6):4663–4678.

OpenAI. 2022. Chatgpt.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang,
Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu,
Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b:
An open bilingual pre-trained model.
arXiv
preprint arXiv:2210.02414.

Tianjun Zhang, Xuezhi Wang, Denny Zhou, Dale
Schuurmans, and Joseph E Gonzalez. 2022. Tem-
pera: Test-time prompting via reinforcement
learning. arXiv preprint arXiv:2211.11890.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye,
and Yanlin Wang. 2024. Memorybank: Enhanc-
ing large language models with long-term memory.
In Proceedings of the AAAI Conference on Artifi-
cial Intelligence, volume 38, pages 19724–19731.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han,
Keiran Paster, Silviu Pitis, Harris Chan, and
Jimmy Ba. 2022. Large language models are
human-level prompt engineers. arXiv preprint
arXiv:2211.01910.

OpenAI. 2024. Gpt-4o.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chen-
guang Zhu, and Michael Zeng. Automatic prompt
optimization with" gradient descent" and beam
In The 2023 Conference on Empirical
search.
Methods in Natural Language Processing.

Taylor Shin, Yasaman Razeghi, Robert L Logan IV,
Eric Wallace, and Sameer Singh. 2020. Auto-
prompt: Eliciting knowledge from language mod-
els with automatically generated prompts. arXiv
preprint arXiv:2010.15980.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao,
Abu Awal Md Shoeb, Abubakar Abid, Adam
Fisch, Adam R Brown, Adam Santoro, Aditya
Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond
the imitation game: Quantifying and extrapolat-
ing the capabilities of language models. arXiv
preprint arXiv:2206.04615.

Rainer Storn and Kenneth Price. 1997. Differen-
tial evolution–a simple and efficient heuristic for
global optimization over continuous spaces. Jour-
nal of global optimization, 11:341–359.

Mirac Suzgun, Nathan Scales, Nathanael Schärli,
Sebastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc V Le, Ed H Chi,
Denny Zhou, et al. 2022. Challenging big-bench
tasks and whether chain-of-thought can solve
them. arXiv preprint arXiv:2210.09261.

Xinyu Tang, Xiaolei Wang, Wayne Xin Zhao, Siyuan
Lu, Yaliang Li, and Ji-Rong Wen. 2024. Unleash-
ing the potential of large language models as
prompt optimizers: An analogical analysis with
gradient-based model optimizers. arXiv preprint
arXiv:2402.17564.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang,
Yann Dubois, Xuechen Li, Carlos Guestrin,
Percy Liang, and Tatsunori B Hashimoto. 2023.
Stanford alpaca: An instruction-following llama
model.

Hugo Touvron, Thibaut Lavril, Gautier Izacard,
Xavier Martinet, Marie-Anne Lachaux, Timothée
Lacroix, Baptiste Rozière, Naman Goyal, Eric
Hambro, Faisal Azhar, et al. 2023. Llama: Open
and efficient foundation language models. arXiv
preprint arXiv:2302.13971.

William Yang Wang. 2017. " liar, liar pants on fire":
A new benchmark dataset for fake news detection.
arXiv preprint arXiv:1705.00648.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanx-
iao Liu, Quoc V. Le, Denny Zhou, and Xinyun
Chen. 2024. Large language models as optimizers.
Preprint, arXiv:2309.03409.

Qinyuan Ye, Maxamed Axmed, Reid Pryzant,
Prompt engi-
arXiv preprint

and Fereshte Khani. 2023.
neering a prompt engineer.
arXiv:2311.05661.

Dataset Name

Task

Train & Dev

Test

LIAR (Wang, 2017)
BBH-Navigate (Suzgun et al., 2022)
ETHOS (Mollas et al., 2022)
ArSarcasm (Farha and Magdy, 2020)
WebNLG (Gardent et al., 2017)
GSM8K (Cobbe et al., 2021)
WSC (Levesque et al., 2012)

True/False
True/False
True/False
True/False
Language Generation
Integer Generation
Multiple-Choice

3681
96
440
8437
200
200
100

461
144
200
2110
300
300
150

Table 7: Dataset sizes and data splits.

Dataset

License

Source

LIAR (Wang, 2017)
BIG-bench Hard (Suzgun et al., 2022) Apache-2.0

Unknown

https://www.cs.ucsb.edu/~cwilliam/data/liar_dataset.zip
https://github.com/google/BIG-bench (original)
https://github.com/suzgunmirac/BIG-Bench-Hard (reformatted)

GNU GPLv3 https://huggingface.co/datasets/iamollas/ethos

ETHOS (Mollas et al., 2022)
ArSarcasm (Farha and Magdy, 2020) MIT
WebNLG (Gardent et al., 2017)
GSM8K (Cobbe et al., 2021)
WSC (Levesque et al., 2012)

CC BY 4.0
MIT
CC BY 4.0

https://github.com/iabufarha/ArSarcasm
https://github.com/fuzihaofzh/webnlg-dataset
https://github.com/openai/grade-school-math
https://huggingface.co/datasets/ErnestSDavis/winograd_wsc

Table 8: License and Source of the datasets used in this study.

A Additional Details for the Setup

A.1 Tasks and Data Details

We present a summary of the dataset sizes and data split information in Table 7. Table 8 provides
details on the sources and licensing information of the datasets. To the best of our knowledge,
our usage of these datasets aligns with their intended purposes, and the data we utilize do not
contain any personal or sensitive information.
LIAR (Wang, 2017) is an English fake news detection corpus comprising 4,000 statements, each
accompanied by context and lie labels. For our experiments, we adopt the same dataset split as
ProTeGi (Pryzant et al.), utilizing 3,681 instances for training and 461 instances for testing.
BIG-bench Hard dataset (Suzgun et al., 2022) is a subset of the BIG Bench dataset (Srivastava
et al., 2022), comprising 23 tasks that present significant challenges for current language models.
For our experiments, we select the navigation task, which requires determining whether an agent,
following a series of navigation steps, returns to its initial starting point. Consistent with the
dataset split used by GPO (Tang et al., 2024), we employ 96 instances for training and 144
instances for testing.
ETHOS (Mollas et al., 2022) is an English hate speech detection dataset consisting of 997 online
comments, each annotated with hate speech labels. In accordance with previous research, we
utilize the same dataset split, employing 440 instances for training and 200 instances for testing.
ArSarcasm dataset (Farha and Magdy, 2020) is an Arabic sarcasm detection corpus containing
10,000 online comments, each labeled for sarcasm. We utilize the original dataset split, with
8,437 instances designated for training and 2,110 instances for testing.
WebNLG corpus consists of sets of triplets that describe facts—entities and the relations
between them—paired with their corresponding expressions in natural language text. Following
the dataset split used by GPO (Tang et al., 2024), we utilize 200 instances for training and 300
instances for testing.
GSM8K (Cobbe et al., 2021) comprises 8.5K high-quality linguistically diverse grade school
math word problems, crafted by human problem writers. Following the dataset split used by
GPO (Tang et al., 2024), we utilize 200 instances for training and 300 instances for testing.
WSC was introduced both as an alternative to the Turing Test and as a measure of a system’s
ability to perform commonsense reasoning. Following the approach used by GPO (Tang et al.,
2024), we sample 100 examples for the training set and 150 for the test set.

Method

LIAR BBH WebNLG

Zero-shot
Five relevant examples
Ours

62.9
65.7
66.6

75.7
78.5
82.6

56.9
57.4
58.8

Table 9: Comparison of our method and dynamically selecting five relevant examples using k-nearest
neighbors (kNN) clustering in the embedding space.

A.2

Implementation Details

We select Doubao-pro (ByteDance, 2024) as the task model and set its temperature to 0, ensuring
deterministic outputs following the GPO (Tang et al., 2024) and AgentInstruct (Crispino et al.,
2023). For the prompt optimizer, we utilize gpt-4o-2024-05-13, the underlying model of GPT-
4o (OpenAI, 2024). Its temperature is set to 1.0 to promote diverse generation. The initial
prompts for different tasks can be found in Section E. In each step, the optimizer generates 8
candidate task prompts. Following GPO (Tang et al., 2024) and OPRO (Yang et al., 2024), the
best-performing one is selected as the task prompt for the next iteration. All experiments are
conducted three times, and we report the average results.

B More Related Work

Prompt engineering aims to identify suitable prompts as inputs for large language models
(LLMs) to perform various tasks. To reduce human effort, researchers have explored automatic
prompt optimization (Lester et al., 2021; Shin et al., 2020; Li and Liang, 2021). Continuous
approaches (Lester et al., 2021; Shin et al., 2020; Li and Liang, 2021) optimize within the
embedding space of LLMs and update based on backpropagating gradients. Prefix tuning (Li and
Liang, 2021) introduces new learnable tokens that can be considered as prompts in continuous
space, which are learned for specific tasks. However, since these tokens are defined in continuous
space, they are not easily interpretable, and these methods require access to model weights,
making them unsuitable for use with closed-source LLMs like ChatGPT (OpenAI, 2022). Discrete
methods (Deng et al., 2022; Zhang et al., 2022) directly optimize natural language prompts.
Several strategies have been developed for this purpose. Some approaches (Pryzant et al.; Juneja
et al., 2024) optimize prompts based on error feedback, while others (Yang et al., 2024; Tang
et al., 2024) utilize multiple prompts and their respective scores to enable the model to identify
superior prompts. Additionally, certain methods (Guo et al., 2024; Fernando et al., 2024; Li
and Wu, 2023) employ genetic algorithms to rewrite prompts through processes of variation and
natural selection. Furthermore, some methods (Ye et al., 2023; Ma et al., 2024) enhance the
controllability of feedback generation and prompt optimization by modifying meta-prompts. To
improve the accuracy of error summaries, some works (Juneja et al., 2024; Austin and Chartock,
2024) cluster similar erroneous samples instead of using randomly selected ones.

C More Ablation Study

Effect of Exemplars’ Solutions. As shown in Table 9, we compared direct retrieval for
prediction on the training set and found that using exemplars yields better results. This is
because (1) the Exemplar Factory pre-assesses exemplars for their effectiveness on the task,
filtering useful ones, and (2) the prompt optimizer crafts chain-of-thought answers tailored to
the questions, enhancing prediction accuracy.

D Meta-Prompt

Here are the meta-prompts we used in Section 3.

I’m trying to write and complete a zero-shot classifier prompt from difficult or erroneous
examples, ‘text’ field means model input, ‘label’ field means true label.
My current prompt is:
{curr_prompt}
But this prompt gets the following examples wrong:
{error_samples}
To improve my understanding and performance, I would like to identify {num_anchor_examples}
typical examples from the above cases where the current prompt fails.
These examples should be diverse to cover a range of different issues.
For each example, provide the following format in JSON and wrap each example with <key_example>
and </key_example>:
<key_example>
{
“text": “{{text}}",
“label": “{{label}}",
“solution": “How to solve this problem step-by-step to get a more accurate answer."
}
</key_example>
After identifying these {num_anchor_examples} typical examples, please provide {num_feedbacks}
reasons why the prompt could have gotten these examples wrong. Wrap each reason with <feedback>
and </feedback>.

Figure 4: Intructive reflection meta-prompt.

I’m trying to write and complete a zero-shot classifier prompt from difficult or erroneous
examples, ‘text’ field means model input, ‘label’ field means true label.
My current prompt is:
{prompt}
But this prompt gets the following examples wrong:
{error_samples}
Based on these examples the problem with this prompt is that:
{feedback}
Based on the above information, I refine the prompt to make the model predict correctly.
The refined prompt is wrapped with <prompt> and </prompt>, less that 512 words:

Figure 5: Optimization meta-prompt.

E Additional Result

Here, we present the initial prompt, the ProTeGi-optimized prompt, and ERM-optimized prompt
across different tasks.

I’m trying to write and complete a zero-shot classifier prompt from difficult or erroneous
examples, ‘text’ field means model input, ‘label’ field means true label.
My current prompt is:
{prompt}
Here are some examples of issues and their labels:
{error_samples}
Here are some suggestions for improving the prompt:
{feedbacks}
Based on the above information, I refine the prompt to make the model predict correctly.
The refined prompt is wrapped with <prompt> and </prompt>, less that 512 words:

Figure 6: Retrieval optimization meta-prompt.

Initial Prompt (Empty)
## Task
Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other
information.
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 7: Initial prompt of the LIAR dataset.

ProTeGi Optimized Prompt
## Task
Evaluate the Statement below using the provided Context and ascertain its factual accuracy (Yes,
it is false or misleading) or accuracy (No, it is not false or misleading). Follow these steps
for your evaluation:
1. Confirm the factual accuracy of the Statement by referencing the given Context and relevant
background information.
2. Take into account the job title, state, and political affiliation of the speaker to gauge
their perspective and potential bias.
3. Assess the plausibility and logical coherence of the Statement.
4. Verify the Statement against established facts and data as necessary.
5. Evaluate whether the Statement, even if factually accurate, is presented in a misleading or
hyperbolic manner.
Statement: A study of private bail bond systems showed that Wisconsin has a higher no-show rate
than other states of defendants skipping court appearances.
Job title: Wisconsin Assembly speaker
State: Wisconsin
Party: Republican
Context: an interview
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 8: ProTeGi optimized prompt of the LIAR dataset.

**Be mindful of hyperbolic, rhetorical, or satirical elements**.

ERM Optimized Prompt
## Task
You are tasked with determining the factual accuracy of statements based on their content,
context, and widely accepted facts. Your goal is to decide whether the statement is false
("Yes") or true ("No"). For each example, you will be provided with:
1. **Statement**: The statement to be evaluated.
2. **Job title**: The job title of the person who made the statement (if available).
3. **State**: The state associated with the person who made the statement (if available).
4. **Party**: The political party of the person who made the statement (if available).
5. **Context**: The situation in which the statement was made, including any relevant background
information.
Instructions:
1. **Evaluate the statement** based on verifiability and supporting evidence from **multiple
reliable sources**.
2. **Cross-reference the statement** with verifiable data and widely accepted facts.
3. **Consider the context** in which the statement was made, including legislative, historical,
and situational nuances.
4. **Ignore the political affiliation** and focus solely on the factual accuracy of the statement.
5. **If a statement is vague, lacks concrete details, or cannot be verified with reliable sources,
answer "Yes."**
6. **If a statement is partially true but omits crucial context or presents facts misleadingly,
answer "Yes."**
7. **If a statement is true and well-supported by reliable evidence, answer "No."**
8. **Pay special attention** to statements with mixed truths; if any part of the statement is
misleading, answer "Yes."
9. **If a statement is statistically accurate but requires nuanced interpretation or context to
be fully understood, answer "No."**
If the core factual
10.
content is accurate and verifiable, answer "No." If hyperbole or rhetoric leads to a misleading
impression, answer "Yes."
11. **Prioritize factual accuracy** and ensure your decision is based on concrete evidence and
context.
12. **For statements with mixed or nuanced truths**, focus on whether the core message is
accurate. If the core message is misleading or omits critical context, answer "Yes." If the core
message is accurate despite requiring nuanced interpretation, answer "No."
Example 1:
- Statement: "Every 28 hours an unarmed black person is shot by a cop."
- Job title: Activist
- State: California
- Party: none
- Context: a speech at a rally
- Answer: Yes
Example 2:
- Statement: "Congressman Renacci is under FBI investigation."
- Job title: Politician
- State: Ohio
- Party: republican
- Context: a news interview
- Answer: Yes
Example 3:
- Statement: "You can’t build a Christian church in Saudi Arabia."
- Job title: Radio/TV host
- State:
- Party: none
- Context: a broadcast on the Sean Hannity radio show
- Answer: No
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 9: ERM optimized prompt of the LIAR dataset.

Initial Prompt (Empty)
## Task
If you follow these instructions, do you return to the starting point?
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 10: Initial prompt of the BBH dataset.

ProTeGi Optimized Prompt
## Task
If you follow these instructions, do you return to the starting point?
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 11: ProTeGi optimized prompt of the BBH dataset.

ERM Optimized Prompt
## Task
You are provided with a sequence of movement instructions. Your objective is to determine if
following these instructions will bring you back to the starting point. Consider every movement
and turn mentioned, including steps to the left, right, forward, and backward. The directive
"Always face forward" implies maintaining your original direction unless specified to turn.
Accumulate the total effect of all movements and turns to determine the final position. The
possible results are:
- Yes
- No
Consider these examples:
1. Instructions: Always face forward. Move 7 steps forward. Move 7 steps backward. Move 10
steps backward. Move 10 steps forward.
Answer: Yes
2. Instructions: Move 8 steps. Move 1 step. Turn left. Turn left. Move 9 steps.
Answer: Yes
3. Instructions: Move 4 steps. Move 2 steps. Turn around. Move 6 steps.
Answer: Yes
4. Instructions: Move 3 steps. Turn left. Turn left. Turn around. Move 5 steps. Turn around.
Move 5 steps.
Answer: No
5. Instructions: Move 6 steps. Move 10 steps. Move 10 steps. Move 1 step. Move 7 steps. Move 5
steps. Turn around.
Answer: No
Now, review the following instructions and determine if you will return to the starting point:
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 12: ERM optimized prompt of the BBH dataset.

Initial Prompt (Empty)
## Task
Is the following text hate speech?
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 13: Initial prompt of the ETHOS dataset.

Please determine whether the provided text constitutes hate speech.

ProTeGi Optimized Prompt
## Task
Input:
Hate speech
encompasses any expression that belittles, discriminates against, or dehumanizes individuals
or groups based on attributes such as race, religion, ethnic background, sexual orientation,
disability, or gender. Analyze if the text contains harmful stereotypes, derogatory language,
or promotes violence or discrimination, including subtle, coded, sarcastic, or ironic language.
Take into account the context and any potentially harmful intentions behind the words, even if
they are not overtly stated. Pay particular attention to implicit forms of hate speech that may
not be explicitly offensive but still perpetuate harmful stereotypes or intentions.
Examples:
- "Men race is a subordinate of women": Yes, this is hate speech.
- "I love deaf people. I can scream on them and I can scold them": Yes, this is hate speech.
- "Don’t hit Women. Just fuck them and make them pregnant.": Yes, this is hate speech.
- "Autists is why presidents like trump are getting elected": Yes, this is hate speech.
Now, please evaluate the following text:
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 14: ProTeGi optimized prompt of the ETHOS dataset.

ERM Optimized Prompt
## Task
You are an expert in identifying hate speech. Hate speech includes any expression, gesture,
conduct, writing, or display that incites violence or prejudicial action against a particular
individual or group or disparages or intimidates a protected individual or group. This includes
derogatory comments about race, religion, ethnicity, sexual orientation, disability, gender,
nationality, medical conditions (such as autism), and more. Hate speech can target individuals
or entire communities and can include threats, slurs, dehumanizing language, and claims of
superiority or inferiority of one gender or group over another.
Given the following text, determine whether it qualifies as hate speech or not. Answer "Yes" if
it is hate speech and "No" if it is not.
Consider the context, implicit meanings, and any underlying biases or derogatory generalizations
in the text. Pay attention to any language that may incite violence, express hate, promote
discrimination, or dehumanize specific groups, including nationalities, religions, disabilities,
medical conditions, or other protected characteristics. Be sensitive to sarcasm, coded language,
indirect expressions of hate, and claims of gender superiority or inferiority. Hate speech can
be directed at any group, including white people and women. Remember to include exclusionary
practices, ableist language, and derogatory comments about medical conditions as potential hate
speech.
*** Note: ** Scientific debates or discussions that do not contain derogatory language or
intentions to harm specific groups should not be considered hate speech. Focus on identifying
language that explicitly or implicitly promotes hate, violence, or dehumanization.
Examples of hate speech include but are not limited to:
- Racist slurs or statements promoting racial superiority.
- Comments that dehumanize or demean individuals based on their religion.
- Statements advocating violence or harm against a specific group.
- Derogatory terms or phrases targeting sexual orientation.
- Ableist language such as "retard," "mentally ill," or any other term that demeans disabilities.
Examples:
Text: "Transgenderism is a DISEASE. Transgenders are an abomination. Not only that, but they’re
also lying to themselves. You CANNOT change your DNA from male to female, or from female to
male. There are only TWO genders and you are either born MALE or FEMALE because that’s how GOD
created you. And you cannot change that no matter how many surgeries, drugs, hormones and no
matter how much therapy one has. FACT."
Label: "Yes"
Text: "Transgenders have to protect each other, not fight each other."
Label: "No"
Text: "Gayness should be removed from the face of the earth."
Label: "Yes"
...
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 15: ERM optimized prompt of the ETHOS dataset.

Initial Prompt (Empty)
## Task
Is this tweet sarcastic?
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 16: Initial prompt of the ArSarcasm dataset.

ProTeGi Optimized Prompt
## Task
Kindly assess the provided tweet to determine if it uses sarcasm. Consider the cultural nuances,
linguistic cues, and overall tone in your evaluation. Offer a comprehensive explanation of your
findings:
Text: "<arabic sentences not supported for display>"
Conclusion: Yes. The phrase "<arabic sentences not supported for display>" (which translates
to "a sheep against the enemy, a lion against the elderly and children") employs sarcasm to
critique someone for showing courage only towards those who are vulnerable.
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 17: ProTeGi optimized prompt of the ArSarcasm dataset.

ERM Optimized Prompt
## Task
Analyze the following tweet to determine if it is sarcastic. Sarcasm often involves saying
the opposite of what one means and may contain elements of irony, exaggeration, mockery, or
complex emotional undertones. Carefully consider the context, including cultural, political,
and social references, which can carry implicit sarcastic undertones in Arabic tweets. Examine
the tweet for subtle clues such as understatement, dry humor, and nuanced emotional tone that
could indicate sarcasm.
Key points to consider:
- **Exaggeration:** Look for statements that sound overly dramatic or extreme.
- **Irony:** Identify instances where the intended meaning is the opposite of the literal wording.
- **Contradictory Statements:** Detect inconsistencies within the tweet itself.
- **Cultural, Political, and Social Nuances:** Recognize idioms, cultural references, and
politically or socially charged statements that suggest sarcasm.
- **Emotional Tone:** Pay attention to signals like bitterness, frustration, mockery, or
exaggerated enthusiasm, which are key indicators of sarcasm.
- **Subtle Clues:** Look for understated comments, dry humor, or nuanced emotional expressions
that may indicate sarcasm. This includes seemingly positive statements with a negative context
or vice versa, and overly enthusiastic remarks that may carry an underlying negative sentiment.
Examples to guide your analysis:
1. "<arabic sentences not supported for display>" – Yes
2. "<arabic sentences not supported for display>" – No
3. "<arabic sentences not supported for display>" – Yes
Now, decide if the given tweet is sarcastic and answer with either "Yes" or "No".
## Output format
Answer Yes or No as labels.
## Prediction
Text: {input}
Label:

Figure 18: ERM optimized prompt of the ArSarcasm dataset.

Initial Prompt (Empty)
## Task
Write the following triples as fluent English text.
## Prediction
{input}
Answer:

Figure 19: Initial prompt of the WebNLG dataset.

ProTeGi Optimized Prompt
## Task
You are given a set of triples that need to be converted into coherent and fluent English
sentences. Each triple consists of a subject, predicate, and object. Your task is to accurately
convey the information from these triples into well-formed sentences. Ensure that the sentences
are complete, grammatically correct, and clearly express the relationships provided in the
triples.
Guidelines:
1. Combine related triples into a single sentence where appropriate.
2. Use synonyms and variations to avoid repetition, but ensure the meaning remains clear and
accurate.
3. Incorporate all relevant information for each subject within the same sentence or group of
sentences.
4. Maintain the context and coherence of the information while ensuring the sentences flow
naturally.
5. Be mindful of the sequence of information to enhance readability and understanding.
6. Clearly differentiate between simple and more complex relationships to fully capture the
depth of the information provided. Pay particular attention to hierarchical relationships or
ownership, clearly distinguishing between entities such as manufacturers, subsidiaries, and
divisions.
## Prediction
{input}
Answer:

Figure 20: ProTeGi optimized prompt of the WebNLG dataset.

ERM Optimized Prompt
## Task
Convert the following triples into coherent and fluent English sentences.
Ensure that all
relationships and attributes are accurately conveyed. When multiple associations or attributes
are involved, break down the information into smaller, logical sentences to maintain clarity.
Example 1:
Triples:
Anders_Osborne | associatedBand/associatedMusicalArtist | Billy_Iuso
Anders_Osborne | associatedBand/associatedMusicalArtist | Tab_Benoit
Anders_Osborne | genre | Rock_music
Anders_Osborne | associatedBand/associatedMusicalArtist | Galactic
Output:
Rock musician Anders Osborne has worked with the band Galactic and also with the musical artists
Tab Benoit and Billy Iuso.
Example 2:
Triples:
Twilight_(band) | genre | Black_metal
Aaron_Turner | associatedBand/associatedMusicalArtist | Twilight_(band)
Aaron_Turner | associatedBand/associatedMusicalArtist | House_of_Low_Culture
Aaron_Turner | instrument | Electric_guitar
Black_metal | musicFusionGenre | Death_metal
Output:
Aaron Turner plays the electric guitar and performed with Twilight, a black metal band, and
House of Low Culture. Black metal is an element of the fusion genre death metal.
Example 3:
Triples:
Baked_Alaska | mainIngredient | "Meringue, ice cream, sponge cake or Christmas pudding"
Baked_Alaska | country | "France, United States or China"
Baked_Alaska | region | "Paris, New York or Hong Kong"
Baked_Alaska | ingredient | Meringue
...
Output:
Baked Alaska has the main ingredients of meringue, ice cream, and sponge cake (or Christmas
pudding). It is found in France, the US, China, Hong Kong, New York, and Paris.
## Prediction
{input}
Answer:

Figure 21: ERM optimized prompt of the WebNLG dataset.

Initial Prompt (Empty)
## Task
Solve the math problem.
## Prediction
Text: {input}
Label:

Figure 22: Initial prompt of the GSM8K dataset.

ProTeGi Optimized Prompt
## Task
Read the following problem carefully and perform the necessary mathematical calculations to
find the correct numerical answer.
## Prediction
Text: {input}
Label:

Figure 23: ProTeGi optimized prompt of the GSM8K dataset.

ERM Optimized Prompt
## Task
Approach this problem methodically by following these steps:
1. **Interpretation:** Carefully read and interpret the problem statement. Pay close attention
to the relationships, conditions, constraints, and sequence of events described. Identify key
quantities and their interrelationships.
2. **Break Down:** Break the problem into manageable steps. Identify the calculations required
for each step and the sequence in which to perform them. Ensure you understand how the different
parts of the problem connect.
3. **Calculation:** Perform the calculations step-by-step. Ensure that each calculation is based
on the correct interpretation of the problem’s conditions. Be meticulous with numerical values,
units, and any given constraints.
4. **Verification:** Double-check your calculations. Verify that each step logically follows
from the previous one and that the final result makes sense in the context of the problem.
Recalculate if necessary to ensure accuracy.
Refer to the following examples for guidance:
Example 1:
Text: "At the burger hut, you can buy a burger for $5, french fries for $3, and a soft drink for
$3. If you order a special burger meal, you get all 3 of these food items for $9.50. A kid’s
burger is $3, a kid’s french fries are $2, and a kid’s juice box is $2. They also have a kids
meal of all 3 kids’ food items for $5. Mr. Parker buys 2 burger meals for his wife and himself.
He also buys 2 burger meals and 2 kid’s meals for his 4 children. How much money does Mr. Parker
save by buying the 6 meals versus buying the individual food items?"
Solution Steps:
- Calculate the individual cost of each adult meal: $5 + $3 + $3 = $11.
- Total cost for 4 adult meals: 4 * $11 = $44.
- Calculate the cost of each kid’s meal: $3 + $2 + $2 = $7.
- Total cost for 2 kids’ meals: 2 * $7 = $14.
- Total cost without meal deals: $44 + $14 = $58.
- Cost with meal deals: 4 * $9.50 (adult meals) + 2 * $5 (kids’ meals) = $38 + $10 = $48.
- Total savings: $58 - $48 = $10.
Label: 10.
Example 2:
Text: "Liam wants to go to Paris, but first, he has to pay his bills. His trip costs $7,000,
and his bills cost $3,500. Knowing that Liam has saved $500/month for 2 years, how much money
will he have left after paying his bills?" Solution Steps:
- Total savings: $500 * 24 months = $12,000.
- Total expenses (trip + bills): $7,000 + $3,500 = $10,500.
- Money left after expenses: $12,000 - $10,500 = $1,500.
Label: 1500.
Example 3:
Text: "Steve has a bank account that earns 10% interest every year. He puts $100 in it, and then
$10 each year. How much money is in it after two years?"
Solution Steps:
- First year: $100 * 1.10 + $10 = $120.
- Second year: $120 * 1.10 + $10 = $142.
Label: 142.
Use these examples as a guide to solve your problem. Carefully verify each step, consider any
numerical variations, and ensure all calculations align with the problem’s conditions. Once you
have your solution, review it to confirm its validity in the context of the problem.
## Prediction
Text: {input}
Label:

Figure 24: ERM optimized prompt of the GSM8K dataset.

Initial Prompt (Empty)
## Task
Solve the problem.
## Prediction
Text: {input}
Label:

Figure 25: Initial prompt of the WSC dataset.

ProTeGi Optimized Prompt
## Task
Carefully read the provided text and identify the entity that the pronoun in the text refers
to. Take into account the context, including relationships and actions described. Select the
correct option (A or B) that corresponds to the referent of the pronoun.
For instance:
- Examine actions that might indicate which entity is being referred to.
- Consider the logical flow of events.
- Notice descriptions and the relative positioning of the entities.
Text: "The sack of potatoes had been placed below the bag of flour, so it had to be moved first.
What does the pronoun "it" refer to?
(A) The sack of potatoes
(B) The bag of flour"
Answer: (B)
Text: "George got free tickets to the play, but he gave them to Eric, because he was particularly
eager to see it. What does the pronoun "he" refer to?
(A) George
(B) Eric"
Answer: (B)
Text: "It was a summer afternoon, and the dog was sitting in the middle of the lawn. After
a while, it got up and moved to a spot under the tree, because it was cooler. What does the
pronoun "it" refer to?
(A) The dog
(B) The spot under the tree"
Answer: (B)
Text: "The sculpture rolled off the shelf because it wasn’t level. What does the pronoun "it"
refer to ?
(A) The sculpture
(B) The shelf"
Answer: (B)
## Prediction
Text: {input}
Label:

Figure 26: ProTeGi optimized prompt of the WSC dataset.

ERM Optimized Prompt
## Task
You will be given a sentence or a pair of sentences containing one or more pronouns. Your task is
to identify the noun or noun phrase that each pronoun most logically refers to, based on context,
causality, descriptive details, and common sense reasoning. Carefully analyze the sentences
and use your understanding of typical human behavior, relationships, and world knowledge to
determine the correct antecedent for each pronoun.
Consider the following guiding principles:
1. **Influence and Causality**: Who or what is causing an action or effect?
2. **Descriptive Context**: What descriptive details precede or follow the pronoun?
3. **Actions and Reactions**: Who is performing or receiving an action?
4. **Contextual Dependencies**: Use background knowledge and the usual roles in interactions to
resolve pronouns accurately.
Examples:
1. "Steve follows Fred’s example in everything. He influences him hugely. What does the pronoun
’He’ refer to?"
(A) Steve
(B) Fred
Answer: (B) Fred
2. "Pete envies Martin because he is very successful. What does the pronoun ’he’ refer to?"
(A) Pete
(B) Martin
Answer: (B) Martin
3. "Sid explained his theory to Mark but he couldn’t convince him. What does the pronoun ’he’
refer to?"
(A) Sid
(B) Mark
Answer: (A) Sid
4. "The fish ate the worm. It was tasty. What does the pronoun ’It’ refer to?"
(A) The fish
(B) The worm
Answer: (B) The worm
Analyze each sentence and select the option that best fits the context and your general knowledge.
## Prediction
Text: {input}
Label:

Figure 27: ERM optimized prompt of the WSC dataset.


