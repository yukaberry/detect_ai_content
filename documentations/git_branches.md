
# Git Branches

We tried to implement a [gitflow principle](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

## master
- The root branch of the project.
- If you want to create a feature, you have to create a branch starting by this one.
- If you want to share your code with other, you have to open a [Pull Request on github](https://github.com/yukaberry/detect_ai_content/compare) and to target this one. (base:master, compare:your_branch)


## production (not yet use)
- The code on production.
- If we want to fix the production, we can fix on that branch.
- If we want to share that fix to master, we have to create a Pull Request to master


## feature/{name}_{your_feature}
- We do development on one specific branch
