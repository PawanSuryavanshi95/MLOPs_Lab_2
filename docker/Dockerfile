FROM python:3.8.1

COPY ./requirements.txt /exp/requirements.txt

RUN pip3 install --no-cache-dir -r /exp/requirements.txt

COPY ./api/*.py /exp/api/

COPY ./svm_gamma=0.001_C=5.joblib /exp/

WORKDIR /exp

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["./api/app.py"]

#ENTRYPOINT ["python", "./api/app.py" ]