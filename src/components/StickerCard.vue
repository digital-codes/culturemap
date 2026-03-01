<template>
    <div class="card-container" ref="cardContainer" :style="{ backgroundImage: `url(${props.cardImage})` }">
    <!-- <img :src="cardImage" ref="cardImageRef" class="image" @load="onImageLoad"/> -->
        <button @click="$emit('close')" class="close">X</button>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const props = defineProps({
    cardImage: {
        type: String,
        required: true,
    }
});

const emit = defineEmits(['rectangle-clicked']);

const cardContainer = ref(null);
const cardImageRef = ref(null);
const imageWidth = ref(0);
const imageHeight = ref(0);

onMounted(() => {
    if (cardImageRef.value) {
        console.log('Card image element found, loading image:', props.cardImage);   
        imageWidth.value = cardImageRef.value.naturalWidth;
        imageHeight.value = cardImageRef.value.naturalHeight;
        console.log('Card image loaded with dimensions:', imageWidth.value, imageHeight.value);
    }
});


</script>

<style scoped>
.card-container {
    position: relative;
    display: inline-block;
    height:100%;
    width: 100%;   
    /* background-image: url("/img/card/3_7d_ovl.png");*/
    background-size: contain;
    background-repeat: no-repeat;
    background-position-x: center;
}

.close {
    /* 
    max-width: 100%;
    height: auto;
    */
    position: absolute;
    top:0;
    left:0;
    width: 3rem;
    height:3rem;
    z-index: 20;
}

.image {
    /* 
    max-width: 100%;
    height: auto;
    */
    height: 100%;
    width: auto;
    margin-left:auto;
    margin-right:auto;
    position: absolute;
    top:0;
    left:0;
    /*
    height:100%;
    width: auto;
    margin-left:auto;
    margin-right:auto;
    */
}
</style>
