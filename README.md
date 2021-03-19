# economic-forecasting
My home on Git for predictive modeling of the U.S. economy

## Project breakdown

As an applied economist one of my long standing on and off projects has been to forecast the short run trajectory of the U.S. economy as measured by gross domestic product. To accomplish this I have used an ensemble of two vector auto regression or var models. This is a linear based technique that involves taking a number of related factors and using them to form predictions of each other once each variable in the model has been predicted for one period out, these predictions are then treated as the most recent period of data to repeat the process to generate predictions for the second period into the future, and so on and so forth until we have predicted as far into the future as we intend to. Traditionally I have done this using the statistical analysis program Stata, however having now transitioned to using Python as my primary analytical environment with my shift into the field of data science, I wanted to recreate this process in Python. 

There are several advantages to this; first I don’t have to switch programs from Python to Stata if I want to utilize this forecasting technique – I can continue to make Python my one stop shop for working with data. Second, Python is much more flexible than Stata as it is a programming language in and of itself and isn’t limited to the real of data analysis. This allows me to much more easily write custom code to automate complex processes. The final reason of course is simply because I can, so why not? 

There are several major steps involved in the forecasting process I use in this project and I’ll discuss my methods for each of them, as well as ways in which this process changed with the utilization of Python here. These general steps are; model specification, error measurement and tuning, Calibration of forecast intervals, and graphing/reporting results. 

## Modeling

This step was largely done previously and so didn’t take any effort this time around as I didn’t make significant changes to the model specifications that I used when conducting this analysis in the past using Stata. I will explain the two methods that I use here, for those who have not read my past blog posts on the subject. 

Essentially there are two different broad methods for forecasting a countries GDP. The first is to forecast it outright as a single variable based on past values of GDP itself as well as other factors
that tend to be good indicators. The second is to forecast the different elements of an economy that are aggregated in the calculation of a country’s GDP. These pieces are consumer spending, government spending, investment spending, and net exports. Instead of choosing one or the other, I have long made it my practice to generate a forecast using each of these methods and then combining the results of the two models by way of weighted averaging to form my final output. 

Between the two models that I use, I incorporate data on the GDP variables in question as well as measures of overall economic health such as employment, industrial production, energy utilization, the housing market and indicators of investment health. Each model has a different mix of explanatory variables which further serves to create a bit of distinction between their outputs. This is useful preventing an overfitting problem as can be done with a single model, and while there certainly aren’t enough outputs being averaged to get the same benefit in this way that a random forest model achieves over a single decision tree, it is still more useful than relying on a single model specification. 

