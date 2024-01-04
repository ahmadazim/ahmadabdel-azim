---
layout: page
title: Disease Risk Prediction
description: Developing risk prediction frameworks for correlated data. <br> Ongoing @ Lin Lab
img: assets/img/projects/project2/cover.png
importance: 1
category: research
---

<p align="center">
<b>Joint estimation of millions of genetic variants to account for correlated observations in Biobank-scale data for predicting disease risk and progression.</b>
</p>

<br />

### Summary

The development of genome-wide polygenic risk scores (PRS) enables early disease diagnosis, careful patient stratification, and genomically-enhanced treatment plans. Despite this clear clinical need for accurate risk assessment, an open question is how to correctly account for correlated data due to familial relatedness and longitudinal health records in training cohorts, for example as a results of familial relatedness or longitudinal observation. Given that there is a lack of strategies available to efficiently derive disease onset and progression insights from correlated data, here we introduce a statistically rigorous risk prediction framework to solve this problem. We hypothesize that through mixed-effects modeling, we can account for correlation among observations and derive more accurate disease risk and even progression insights. We validate our approach through simulation and apply it to Biobank data, and we argue that it is indeed time to include polygenic risk prediction in the clinical setting.

In particular, we have the following aims:
- Introduce a unified statistical framework for polygenic risk score prediction in the presence of correlated data, leveraging mixed effects modeling. 
- Evaluate the performance of this new approach via simulation across several genetic architectures. Specifically, we will simulate related cohorts and longitudinal clinical records and compare the performance and accuracy of our approach to that of existing methods. 
- Apply our approach to compute PRS for common complex diseases in the UK Biobank and demonstrate the potential of our method for clinical translation of polygenic prediction.

