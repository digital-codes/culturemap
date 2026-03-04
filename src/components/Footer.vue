<script setup lang="ts">
import gh from "@primer/octicons/build/svg/mark-github-16.svg"

</script>
<template>
    <div class="footer">
        <hr>
        <p>{{ $t('message.stay tuned') }}</p>
        <p class="sourcecode">{{ $t('message.sourceCode') }} <a href="https://github.com/digital-codes/culturemap" target="_blank"><img :src="gh" alt="GitHub Logo" class="gh" ></a></p>
        
    </div>
</template>

<style scoped>
.sourcecode {
    font-family: "Audiowide", cursive;
    text-align: center;
    color: var(--color-text);
    display:inline-flex;
    justify-content: center;
}   
.gh {
    margin: 0;
    margin-left: 0.2rem;
    margin-right: 0.2rem;
    width: 1.5rem;
    height: 1.5rem;
    background-color: white;
}
</style>
