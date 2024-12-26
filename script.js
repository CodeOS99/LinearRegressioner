let prec = 3;
let chartInstance = null;

document.getElementById('submit').addEventListener('click', function() {
    const fileInput = document.getElementById('training');
    const testing = document.getElementById('testing');
    const alpha = document.getElementById('alpha').value;
    const numIters = document.getElementById('numIters').value;
    const writing = document.getElementById('writing').value;
    const verbose = document.getElementById('verbose').checked;
    const nvl = document.getElementById('nvl').checked;

    const trainingFile = fileInput.files[0];
    const testingFile = testing.files[0];
    console.log(testingFile)

    let data;
    let testingData;
    let flag = 0;
    if (testingFile && trainingFile) {
        const reader = new FileReader();
        reader.onload = function (event) {
            console.log(("ued"),flag)
            // `event.target.result` gives the raw content of the CSV file
            const lines = event.target.result.split('\n');  // Split the file into lines
            if(flag === 0) {
                data = lines.map((z) => z.split(','));  // Split each line into values by comma
                flag++;
                reader.readAsText(testingFile);  // Read the file as text
            } else {
                testingData = lines.map((z) => z.split(','));  // Split each line into values by comma
                flag++;
            }
            if(flag === 2) {
                document.getElementById("results").innerHTML = ''
                performLinearRegression(data,alpha,numIters,writing,testingData,verbose,nvl);
            }
        };
        
        reader.readAsText(trainingFile);  // Read the file as text
    } else {
        alert('Please upload a CSV file for each the testing and training data.');
    }
});

function calculatePrediction(w,b,x) {
    let res = 0;
    for(let j = 0; j < x.length; j++) {
        res += w[j]*x[j];
    }
    res += b;
    return res;
}

function calculateLoss(w,b,x,y) {
    loss = 0;
    for(let i = 0; i < x.length; i++) {
        yHat = calculatePrediction(w,b,x[i]);
        loss += (yHat-y[i])**2
    }
    loss /= 2*x.length;
    return loss;
}

function calculateGradient(w,b,x,y,m,n) {
    let dj_dw = new Array(n).fill(0);
    let dj_db = 0;

    for(let i = 0; i < x.length; i++) {
        yHat = calculatePrediction(w, b, x[i]);
        let diff = yHat-y[i];
        dj_db += diff
        for(let j = 0; j < x[0].length; j++) {
            dj_dw[j] += diff * x[i][j];
        }
    }
    dj_dw = dj_dw.map((a) => a/=m);
    dj_db /= m;

    return [dj_dw, dj_db];
}

function gradientDescent(initW, initB, x, y, m, n, alpha, numIters, writing, nvl) {    
    let w = initW;
    let b = initB;
    let losses = [];

    for (let i = 0; i < numIters; i++) {
        [dj_dw, dj_db] = calculateGradient(w, b, x, y, m, n);
        for (let j = 0; j < w.length; j++) {
            w[j] -= alpha * dj_dw[j];
        }
        b -= alpha * dj_db;

        let loss = calculateLoss(w, b, x, y);
        losses.push(loss);

        if (i % writing === 0) {
            document.getElementById("results").innerHTML += `Iteration ${i + 1}, loss - ${loss.toFixed(prec)}<br>`;
        }
    }

    if(nvl === true) {
        plotLossGraph(losses);
    } else {
        if(chartInstance !== null) {
            chartInstance.destroy()
        }
    }
    return [w, b];
}


// black magic
function plotLossGraph(losses) {
    if (chartInstance) {
        chartInstance.destroy();
    }

    const ctx = document.getElementById('lossChart').getContext('2d');

    chartInstance = new Chart(ctx, {
        type: 'line', 
        data: {
            labels: Array.from({ length: losses.length }, (_, i) => i + 1), // Iterations (1, 2, 3, ...)
            datasets: [{
                label: 'Loss',
                data: losses,
                borderColor: 'rgba(75, 192, 192, 1)',
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Loss vs Iterations'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Iteration'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                }
            }
        }
    });
}

function performLinearRegression(data,alpha,numIters,writing,testing,verbose,nvl) {
    console.log(verbose);
    let m = data.length; // number of rows
    let n = data[0].length; // number of columns

    let w = Array(n).fill(0);
    let b = Math.random();
    
    let x = [];
    let y = [];

    let x_test = [];
    let y_test = [];
    
    // For training data
    for (let i = 0; i < m; i++) {
        let row = data[i];
        let features = row.slice(0, n - 1); // all columns except last
        x.push(features);
        if(i !== 0) {
            let target = row[n - 1]; // last column as target
            y.push(target.replace(/(\r\n|\n|\r)/gm, ""));
        }
    }

    // For testing data
    for (let i = 0; i < testing.length; i++) {
        let row = testing[i];
        let features = row.slice(0, n - 1); // all columns except last
        x_test.push(features);
        if(i !== 0) {
            let target = row[n - 1]; // last column as target
            y_test.push(target.replace(/(\r\n|\n|\r)/gm, ""));
        }
    }    
    x.shift();
    x_test.shift();

    //let x_norm = normalize(x);
    //let x_test_norm = normalize(x_test);
    
    let x_norm = x;
    let x_test_norm = x_test;

    [w, b] = gradientDescent(w,b,x_norm,y,m,n,alpha,numIters,writing,nvl);

    document.getElementById("results").innerHTML += `<h3>Finished training!</h3><br><p>Final loss - ${calculateLoss(w,b,x_norm,y).toFixed(prec)}</p><br><h2>Results on testing data:</h2>`

    document.getElementById("results").innerHTML += `<p>The loss on testing set is ${calculateLoss(w,b,x_test_norm,y_test).toFixed(prec)}</p>`

    if(verbose === true) {
        for(let i = 0; i < x_test_norm.length; i++) {
            document.getElementById("results").innerHTML += `<p>The result for input ${x_test[i]} are ${calculatePrediction(w,b,x_test_norm[i]).toFixed(prec)}</p><br>`;
        }
    }
    w.pop(); // black magic v2
    document.getElementById("results").innerHTML += `<p>The weights and biases found through gradient descent are:<br>w: ${w}<br>b: ${b}<br></p>`
}

// This somehow made results worse so its excluded for now
/*function normalize(data) {
    temp = structuredClone(data);
    const n = temp[0].length;  // Number of features
    const m = temp.length;     // Number of examples
    
    const mean = Array(n).fill(0);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            mean[j] += temp[i][j];
        }
    }
    mean.forEach((_, i) => mean[i] /= m);  // Mean for each feature

    const dev = Array(n).fill(0);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            dev[j] += (temp[i][j] - mean[j]) ** 2;
        }
    }
    dev.forEach((_, i) => dev[i] = Math.sqrt(dev[i] / (m - 1)));  // Standard deviation for each feature

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            temp[i][j] = (temp[i][j] - mean[j]) / dev[j];  // Normalize
        }
    }

    return temp;
}
*/