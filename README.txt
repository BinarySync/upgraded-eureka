The creators of this projects are BinarySync, NanDSi and StackOverFlow, thanks.

-------------------------------------------

git init
	<folder-name>: allows user to create a new initializated folder for a git repo/project.

git config
	--global
	user.name <username>
	user.email <email>

git clone <git_link>
	<folder-name>: allows user to clone the repo to a specific and new folder.

git add <file_name>: adds files for commiting

git rm <file_name>: removes files for commiting
 
git commit: makes a commit, a new version of the project
	-m <description>: allows describing directly into arguments
	
git push: sends commit to .git
	-f <git_link> <branch>

git pull: downloads newest version of the project to your folder(if is not updated)

git reset 
	--hard <commit_id>: resets the branch to a speficific version of the branch, allowing row backs.
	
-------------------------------------------

Useful COMBOS:

1. The CLONER Configer:
	git clone <git_link>
	cd <folder_name>
	git config user.name <username>
	git config user.email <email>
	
2. The RollBacker:
	git reset --hard <commit_id>
	git push -f <git_link> master
	
3. The Commiter:
	git add <file_name>	OR	git rm <file_name> 
	git commit -m "Update text, the usual"
	git push
