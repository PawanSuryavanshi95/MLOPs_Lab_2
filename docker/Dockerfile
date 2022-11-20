FROM python:3.9
WORKDIR /exp
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt

COPY ./*.py /exp/

#COPY ./*.sh /exp/

COPY ./api/*.py /exp/api/
COPY ./models/* /exp/models/
COPY ./results/* /exp/results/

#ENTRYPOINT ["python", "-m", "plot_graphs"]

ENTRYPOINT ["python", "./api/app.py" ]