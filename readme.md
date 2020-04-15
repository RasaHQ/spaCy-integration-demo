<img src="square-logo.svg" width=200 height=200 align="right">

# Demo of spaCy in Rasa

This repository contains an example of spaCy in Rasa.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

## Youtube 

This repository is part of a youtube video on spaCy in Rasa. You can watch this video [here]()

## Start 

We've added a `Makefile` that may be helpful (it's commands will be
explained after running `make help`). If you prefer to everything by
hand, you can also run everything by running these commands in the 
terminal. You will need two terminals.

```bash
# terminal one
rasa run actions
# terminal two 
rasa train
rasa shell --endpoints endpoints.yml
```

You can see what the action server receives by looking at the action server logs. 

## Test 

If you want to change the NLU pipeline and see the effect. You can do so via; 

```bash
# before changing `config.yml`
rasa train; rasa test --out before
# after changing `config.yml`
rasa train; rasa test --out after
```

You will now have two folders with results. 

## Play 

Feel free to play around with this! Happy hacking!
