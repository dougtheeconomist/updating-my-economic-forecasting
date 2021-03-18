# economic-forecasting
My home on Git for predictive modeling of the U.S. economy

## Project breakdown

As an applied economist one of my long standing on and off projects has been to forecast the short run trajectory of the U.S. economy as measured by gross domestic product. To accomplish this I have used an ensemble of two vector auto regression or var models. This is a linear based technique that involves taking a number of related factors and using them to form predictions of each other once each variable in the model has been predicted for one period out, these predictions are then treated as the most recent period of data to repeat the process to generate predictions for the second period into the future, and so on and so forth until we have predicted as far into the future as we intend to. Traditionally I have done this using the statistical analysis program Stata, however having now transitioned to using Python as my primary analytical environment with my shift into the field of data science, I wanted to recreate this process in Python. 
There are several advantages to this; first I don’t have to switch programs from Python to Stata if I want to utilize this forecasting technique – I can continue to make Python my one stop shop for working with data. Second, Python is much more flexible than Stata as it is a programming language in and of itself and isn’t limited to the real of data analysis. This allows me to much more easily write custom code to automate complex processes. The final reason of course is simply because I can, so why not? 

There are several major steps involved in the forecasting process I use in this project and I’ll discuss my methods for each of them, as well as ways in which this process changed with the utilization of Python here. These general steps are; model specification, error measurement and tuning, Calibration of forecast intervals, and graphing/reporting results. 

## Modeling
