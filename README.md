*DMA Rankr*
# ---

### This is a novel generative model which ranks where, geographically, it is most opportunistic to promote certain television programs.  [Here](https://github.com/dsrub/A-Generative-Model-for-Promo-Placing-for-TV-Shows-/blob/master/Notesv2.pdf) is a full description of the model and the math behind it.

### The model:

- computes the expected lift in television ratings of a program by geography if it is promoted there
- ranks the geographies by expected lift in order to decide where to promote
- uses a generative model to realistically model ratings in the presence of promos
- uses the [Pomegranate](https://pomegranate.readthedocs.io/en/latest/) API for maximum likelihood estimation using the EM algorithm
- incorporates realistic correlation between viewing behavior in the presence of promos vs. whether promos are not shown
- uses Facebook's [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) API to forecast the number of television viewers when the promotional campaign will occur
- is robust from program-to-program and geography-to-geography

