

--> really high max_token limit ~ 10k
--> increase max_token to death/disch token


----

1. Model config
----

BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001


2. Router Specialization


--> remove expert -- see if it performs worse on certain tasks. 
--> what set of tokens activate networks. 


3. Include everythng that is recorded during admission. 