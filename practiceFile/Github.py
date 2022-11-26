from github import Github

def main():
    Tinput = input("enter your token")
    g=Github(Tinput)
    usr = g.get_user()
    print("user: " + usr.login)
    print("fullname: " + usr.name)
    print("location: " + usr.location)
    print("company: " + usr.company)

main()