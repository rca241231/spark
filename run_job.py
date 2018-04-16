import click

from cluster.cluster import cluster


JOBS = {
    'CLUSTER': 'cluster'
}


@click.command()
@click.option('--survey', help="""
The survey that clustering model is based on.
""")
@click.option('--cluster_output', help="The location to output cluster to.")
@click.option('--centroids_output', help="The location to output centroids to.")
def run_job(**kwargs):
    job = kwargs.get('job')

    if job == JOBS['CLUSTER']:
        cluster(**kwargs)
    else:
        raise ValueError("The job {} is not supported.".format(job))


if __name__ == '__main__':
    run_job()
