The Volterra equations

The interactions between species in an ecosystem is at the heart of the study of ecology. In particular, the predator-prey relation gives rise to quite non-trivial dynamics in their population evolution. The Volterra equations (or Lotka-Volterra) were proposed to model the evolution of the simplest ecosystem, with a predator species and a prey. Taking  𝑦1(𝑡)  and  𝑦2(𝑡)  as the populations of prey and predator, respectively, the Volterra equations are

d𝑦1d𝑡d𝑦2d𝑡==−𝑎𝑦1−𝑏𝑦1𝑦2𝑐𝑦2+𝑑𝑦1𝑦2. 

The positive  𝑎  term quantifies the exponential growth of the prey population in the absence of predators: if  𝑏=0  the first equation decouples from the second and becomes a simple ODE; the fact that the offspring number is proportional to the reproducing population gives rise to an exponential growth. The model assumes unlimited food supply for the prey (in the typical rendering of this problem in terms of hares and foxes, think of enormous carrot fields available to the hares).

The  𝑏  term accounts for the decrease of prey population due to deadly encounters with the predator. The term assumes that such deaths are proportional to the probability of their meeting, which is proportional to the product of their respective populations. The equations also assume that hare deaths by other causes are negligible. The second equation considers the predator population variation to be determined by natural deaths (proportional to the population itself, leading to exponential decay in the absence of prey), except if finding prey, simulated by a similar term as before, with opposite sign of course.

This extremely simple model gives rise to very rich dynamics, due to the fact that it is a non linear model (which also makes it hard for analytic treatment). The description, analysis and ecological implications of the model are beyond the remit of this course, but it serves us nicely to illustrate the power of numerical approaches. NumPy is quite useful here, however, in giving us a very accessible tool for their numerical exploration.
