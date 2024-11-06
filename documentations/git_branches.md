
# Git Branches

We tried to implement a [gitflow principle](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

## master
- The root branch of the project.
- It's the code on Production (FastApi + Streamlit)

## development 
- If you want to create a feature, you have to create a branch starting by this one.
- If you want to share your code with other, you have to open a [Pull Request on github](https://github.com/yukaberry/detect_ai_content/compare) and to target this one. (base:master, compare:your_branch)

## feature/{name}_{your_feature}
- We do development on one specific branch
- Your brnanch should start from **development** branch
