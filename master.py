# request.GET['text_box']
# print("Location:http://newurl.com/foobar")
import naive_bayes

def POST():
    parameters = [request.GET['orbitalPeriod'], request.GET['planetaryRadii'], request.GET['equilibriumTemp'], request.GET['insolationFlux'], request.GET['stellarEffTemp'], request.GET['stellarRadius']]
    after_check = []

    for i in xrange(len(parameters)):
        try:
            after_check.append(float(parameters[i]))
        except ValueError:
            print("Location:http://cool.us-west-2.elasticbeanstalk.com/fail.html")

    label = naive_bayes.main(parameters)

    op = request.GET['orbitalPeriod']
    pr = request.GET['planetaryRadii']
    et = request.GET['equilibriumTemp']

    if label == 1.:
        print("Location:http://cool.us-west-2.elasticbeanstalk.com/superearth.html")
    else:
        if (op < 111) and (pr > 12) and (et > 5000):
            print("Location:http://cool.us-west-2.elasticbeanstalk.com/hotjupiter.html")
        elif (op > 800) and (pr > 12) and (et < 10):
            print("Location:http://cool.us-west-2.elasticbeanstalk.com/gasIceGiant.html")
        else:
            print("Location:http://cool.us-west-2.elasticbeanstalk.com/waterworld.html")
