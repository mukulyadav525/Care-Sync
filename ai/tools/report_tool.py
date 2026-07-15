def create_report(analysis):

    report=[]

    report.append("Health Summary\n")

    for k,v in analysis.items():

        report.append(

            f"{k}: {v}"

        )

    return "\n".join(report)