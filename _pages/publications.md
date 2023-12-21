---
layout: page
permalink: /publications/
title: Publications
description: Publications and manuscripts in press are included below, categorized by field.
years: [Statistical Genetics, Cancer Genomics, Microbiology]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  <div style="height: 50px;"></div>
  {% bibliography -f {{ site.scholar.bibliography }} -q @*[category={{y}}]* %}
{% endfor %}

</div>
