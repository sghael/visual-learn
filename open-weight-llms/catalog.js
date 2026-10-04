(() => {
  const search=document.querySelector('#search'),category=document.querySelector('#category'),order=document.querySelector('#order'),list=document.querySelector('.catalog');
  const items=[...list.children];
  function update(){
    const query=search.value.trim().toLocaleLowerCase();let count=0;
    for(const item of items){const match=(!query||item.textContent.toLocaleLowerCase().includes(query))&&(category.value==='all'||item.dataset.category===category.value);item.hidden=!match;if(match)count++;}
    const sorted=order.value==='newest'?[...items].reverse():items;sorted.forEach(item=>list.append(item));
    document.querySelector('#match-count').textContent=`${count} of ${items.length} tutorials`;
    document.querySelector('#empty').hidden=count>0;
  }
  [search,category,order].forEach(el=>el.addEventListener('input',update));
  document.querySelector('#clear').addEventListener('click',()=>{search.value='';category.value='all';order.value='oldest';update();search.focus();});
  update();
})();
