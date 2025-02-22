// randomUtils.js
// All random-generation helper functions in one place.

export function generateGates() {
    // For forget/input/candidate/output gate
    return {
      forget: Math.random(),
      input: Math.random(),
      candidate: Math.random(),
      output: Math.random()
    };
  }
  
  export function generateRandomOutputValue() {
    // e.g. (Math.random()*2 - 1).toFixed(3)
    return (Math.random() * 2 - 1).toFixed(3);
  }
  
  export function generateRandomWeight(decimals = 3) {
    // For the internal LSTM “connections”
    return (Math.random() * 2 - 1).toFixed(decimals);
  }
  
  export function generateRandomVector(length, decimals = 3) {
    return Array.from({ length }, () => +(Math.random().toFixed(decimals)));
  }
  