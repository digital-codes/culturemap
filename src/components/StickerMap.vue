<template>
    <div class="map-container" ref="mapContainer">
        <img :src="mapImage" @load="onImageLoad" @click="handleMapClick" ref="mapImageRef" />
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const props = defineProps({
    mapImage: {
        type: String,
        required: true,
    },
    rectangles: {
        type: Array,
        required: true,
    },
});

const emit = defineEmits(['rectangle-clicked']);

const mapContainer = ref(null);
const mapImageRef = ref(null);
const imageWidth = ref(0);
const imageHeight = ref(0);

const onImageLoad = () => {
    if (mapImageRef.value) {
        imageWidth.value = mapImageRef.value.naturalWidth;
        imageHeight.value = mapImageRef.value.naturalHeight;
        console.log('Image loaded with dimensions:', imageWidth.value, imageHeight.value);
    }
};

const handleMapClick = (event) => {
    if (!mapImageRef.value) return;
    console.log('Map clicked at:', event.clientX, event.clientY);
    const rect = mapImageRef.value.getBoundingClientRect();
    const scaleX = imageWidth.value / rect.width;
    const scaleY = imageHeight.value / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    console.log('Translated click to image coordinates:', x, y);

    let clickedIndex = -1;

    // Iterate from the end to find the last matching rectangle
    for (let i = props.rectangles.length - 1; i >= 0; i--) {
        const [[bboxX0, bboxY0], [bboxX1, bboxY1]] = props.rectangles[i].bbox;
        console.log(`Checking rectangle ${i} with bbox:`, bboxX0, bboxY0, bboxX1, bboxY1);
        if (
            x >= bboxX0 &&
            x <= bboxX1 &&
            y >= bboxY0 &&
            y <= bboxY1
        ) {
            clickedIndex = i;
            break;
        }
    }

    if (clickedIndex !== -1) {
        emit('rectangle-clicked', clickedIndex);
    }
};
</script>

<style scoped>
.map-container {
    position: relative;
    display: inline-block;
}

.map-container img {
    max-width: 100%;
    height: auto;
}
</style>
